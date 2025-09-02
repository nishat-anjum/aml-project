import argparse
from wav2vec2_interpretability_toolkit import ExplainConfig, explain_one

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, type=str, help="Path to fine-tuned wav2vec2 model directory")
    ap.add_argument("--audio", required=True, type=str, help="Path to a .wav (16kHz mono recommended)")
    ap.add_argument("--out", default="outputs", type=str, help="Output directory to save figures")
    args = ap.parse_args()

    cfg = ExplainConfig(model_path=args.model_path, audio_path=args.audio, out_dir=args.out)
    paths = explain_one(cfg)
    print("Saved files:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
