import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from captum.attr import IntegratedGradients, Saliency, NoiseTunnel


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

    # if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    #     return torch.device("mps")


def load_model_and_extractor(model_path: str, device: Optional[torch.device] = None):
    if device is None:
        device = select_device()
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_path,
        output_attentions=True,
        torch_dtype=torch.float32
    )
    model.to(device)
    model.eval()
    extractor = AutoFeatureExtractor.from_pretrained(model_path)
    return model, extractor, device


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(file_path)
    if wav.dim() == 2:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    else:
        wav = wav.unsqueeze(0)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
        sr = target_sr
    return wav.squeeze(0), sr


def prepare_inputs(waveform: torch.Tensor, extractor, device: torch.device) -> Dict[str, torch.Tensor]:
    inputs = extractor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


@torch.no_grad()
def predict(model, inputs: Dict[str, torch.Tensor]) -> Tuple[int, np.ndarray]:
    logits = model(**inputs).logits
    probs = logits.softmax(dim=-1)
    pred = int(probs.argmax(dim=-1).item())
    return pred, probs.squeeze(0).cpu().numpy()


def forward_for_captum(model, input_values: torch.Tensor) -> torch.Tensor:
    out = model(input_values).logits
    return out


def integrated_gradients(
        model, inputs: Dict[str, torch.Tensor], target: int, n_steps: int = 64,
        internal_batch_size: Optional[int] = None
) -> np.ndarray:
    input_values = inputs["input_values"].clone().detach().requires_grad_(True)  # (1, T)

    ig = IntegratedGradients(lambda x: forward_for_captum(model, x))
    attributions = ig.attribute(
        input_values,
        baselines=torch.zeros_like(input_values),
        target=target,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size
    )
    attr = attributions.squeeze(0).detach().cpu().numpy()
    return attr


def gradient_saliency(
        model, inputs: Dict[str, torch.Tensor], target: int, smooth: bool = True, stdev: float = 0.02,
        nt_samples: int = 25
) -> np.ndarray:
    input_values = inputs["input_values"].clone().detach().requires_grad_(True)

    sal = Saliency(lambda x: forward_for_captum(model, x))
    if smooth:
        nt = NoiseTunnel(sal)
        attributions = nt.attribute(
            input_values,
            target=target,
            nt_samples=nt_samples,
            stdevs=stdev,
            nt_type="smoothgrad"
        )
    else:
        attributions = sal.attribute(input_values, target=target, abs=True)

    attr = attributions.squeeze(0).detach().cpu().numpy()
    return attr


def attention_rollout(attentions: List[torch.Tensor], head_agg: str = "mean",
                      add_residual: bool = True) -> torch.Tensor:
    attn = torch.stack(attentions, dim=0)
    L, B, H, T, _ = attn.shape

    if head_agg == "mean":
        attn = attn.mean(dim=2)
    elif head_agg == "max":
        attn = attn.max(dim=2).values
    else:
        raise ValueError("head_agg must be 'mean' or 'max'")

    I = torch.eye(T, device=attn.device).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
    if add_residual:
        attn = attn + I

    attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    rollout = attn[0]
    for l in range(1, L):
        rollout = torch.matmul(rollout, attn[l])

    rollout_score = rollout.mean(dim=1)  # (B, T)
    rollout_score = rollout_score / rollout_score.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return rollout_score  # (B, T)


@torch.no_grad()
def get_attention_importance(model, inputs: Dict[str, torch.Tensor], upsample_to: int) -> np.ndarray:
    outputs = model(output_attentions=True, **inputs)
    atts = [a for a in outputs.attentions]  # list length L, each (B,H,T,T)
    roll = attention_rollout(atts)  # (B, T)
    roll_upsampled = F.interpolate(roll.unsqueeze(1), size=upsample_to, mode="linear", align_corners=False)
    return roll_upsampled.squeeze(0).squeeze(0).cpu().numpy()


@torch.no_grad()
def prob_for_target(model, inputs: Dict[str, torch.Tensor], target: int) -> float:
    logits = model(**inputs).logits
    probs = logits.softmax(dim=-1)
    return float(probs[0, target].item())


def deletion_curve(
        model, extractor, waveform: torch.Tensor, device: torch.device, attributions: np.ndarray,
        target: int, window: int = 400, step: int = 200
) -> Tuple[np.ndarray, np.ndarray, float]:
    attr = np.abs(attributions).astype(np.float64)
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-12)

    T = attr.shape[0]
    masked = waveform.clone().detach().cpu().numpy().copy()

    importance = np.zeros(T, dtype=np.float64)
    for i in range(0, T - window + 1, step):
        importance[i:i + window] += attr[i:i + window].sum()

    order = np.argsort(-importance)  # descending
    segments = []
    seen = np.zeros(T, dtype=bool)
    for idx in order:
        start = idx
        end = min(idx + window, T)
        if not seen[start:end].any():
            segments.append((start, end))
            seen[start:end] = True

    xs, ys = [0.0], []
    base_inputs = prepare_inputs(waveform, extractor, device)
    base_prob = prob_for_target(model, base_inputs, target)
    ys.append(base_prob)

    current = masked.copy()
    total_removed = 0
    total = float(T)

    for (s, e) in segments:
        current[s:e] = 0.0
        total_removed += (e - s)

        curr_inputs = prepare_inputs(torch.tensor(current), extractor, device)
        p = prob_for_target(model, curr_inputs, target)
        xs.append(min(1.0, total_removed / total))
        ys.append(p)

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    auc = np.trapz(ys, xs)
    return xs, ys, float(auc)


def _normalize(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float64)
    a = (a - a.min()) / (a.max() - a.min() + 1e-12)
    return a


def plot_wave_and_attr(wave: np.ndarray, attr: np.ndarray, title: str, out_path: Optional[str] = None):
    wave_n = _normalize(wave)
    attr_n = _normalize(np.abs(attr))

    plt.figure(figsize=(12, 3))
    plt.plot(wave_n, label="Waveform")
    plt.plot(attr_n, label="Attribution")
    plt.legend()
    plt.title(title)
    if out_path:
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
    plt.close()


def plot_curve(xs: np.ndarray, ys: np.ndarray, title: str, xlabel: str, ylabel: str, out_path: Optional[str] = None):
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if out_path:
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
    plt.close()


@dataclass
class ExplainConfig:
    model_path: str
    audio_path: str
    out_dir: str = "outputs"
    n_steps_ig: int = 64
    saliency_smooth: bool = True
    saliency_stdev: float = 0.02
    saliency_samples: int = 25
    occl_window: int = 400  # ~25ms at 16kHz
    occl_step: int = 200  # 50% overlap


def explain_one(cfg: ExplainConfig) -> Dict[str, str]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    model, extractor, device = load_model_and_extractor(cfg.model_path)
    wave, sr = load_audio(cfg.audio_path, target_sr=16000)
    inputs = prepare_inputs(wave, extractor, device)

    # Prediction
    pred, probs = predict(model, inputs)
    label = int(pred)
    print(f"label: {label}")
    prob = float(probs[label])

    # IG
    ig_attr = integrated_gradients(model, inputs, target=label, n_steps=cfg.n_steps_ig)
    plot_wave_and_attr(
        wave.cpu().numpy(),
        ig_attr,
        f"Integrated Gradients (pred={label}, p={prob:.3f})",
        out_path=os.path.join(cfg.out_dir, "ig.png"),
    )

    # Saliency
    sal_attr = gradient_saliency(
        model, inputs, target=label, smooth=cfg.saliency_smooth,
        stdev=cfg.saliency_stdev, nt_samples=cfg.saliency_samples
    )
    plot_wave_and_attr(
        wave.cpu().numpy(),
        sal_attr,
        "Gradient Saliency (SmoothGrad)" if cfg.saliency_smooth else "Gradient Saliency",
        out_path=os.path.join(cfg.out_dir, "saliency.png"),
    )

    # Attention
    att_attr = get_attention_importance(model, inputs, upsample_to=wave.shape[0])
    plot_wave_and_attr(
        wave.cpu().numpy(),
        att_attr,
        "Attention Rollout (upsampled)",
        out_path=os.path.join(cfg.out_dir, "attention_rollout.png"),
    )

    # Faithfulness via Deletion Curve
    xs, ys, auc = deletion_curve(
        model, extractor, wave, device, ig_attr, target=label,
        window=cfg.occl_window, step=cfg.occl_step
    )
    plot_curve(xs, ys, f"Deletion Curve (AUC={auc:.3f})", "Fraction removed", "Target prob",
               out_path=os.path.join(cfg.out_dir, "deletion_curve.png"))

    summary_txt = os.path.join(cfg.out_dir, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"Audio: {cfg.audio_path}\n")
        f.write(f"Predicted class: {label}\n")
        f.write(f"Probabilities: {probs.tolist()}\n")
        f.write(f"Deletion AUC (IG-based): {auc:.6f}\n")

    return {
        "ig_png": os.path.join(cfg.out_dir, "ig.png"),
        "saliency_png": os.path.join(cfg.out_dir, "saliency.png"),
        "attention_png": os.path.join(cfg.out_dir, "attention_rollout.png"),
        "deletion_png": os.path.join(cfg.out_dir, "deletion_curve.png"),
        "summary_txt": summary_txt
    }
