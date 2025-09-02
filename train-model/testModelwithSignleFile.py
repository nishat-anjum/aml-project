import torch
import os
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
def main():
    processor = AutoFeatureExtractor.from_pretrained("../fine_tuned_model/wav2vec2_bangla_toxic_model")
    model = AutoModelForAudioClassification.from_pretrained("../fine_tuned_model/wav2vec2_bangla_toxic_model")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def predict_single_file(file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        inputs = processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt", truncation=True,
                           padding="max_length", max_length=64000)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = model(input_values)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        return predicted_class

    cwd = os.getcwd()
    file_name = "../test_data/uniq_non-toxic.wav"
    file_path = os.path.join(cwd, file_name)
    predicted_class = predict_single_file(file_path)
    print(f"Predicted class: { 'fine_tuned_model' if predicted_class == 1 else 'non-fine_tuned_model' }")

if __name__=="__main__":
    main()