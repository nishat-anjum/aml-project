# Toxic Audio Detection & Interpretability (Bangla, Wav2Vec2.0)

This project fine-tunes a **Bangla Wav2Vec2.0** model for **toxic vs. non-toxic speech classification** and provides tools for **data preprocessing, training, prediction, and interpretability** using **Integrated Gradients** and **Saliency Maps**.  

---

## Table of Contents
- [Setup](#️-setup)  
- [Data Preprocessing](#-data-preprocessing)  
- [Training the Model](#-training-the-model)  
- [Running Training Script Locally](#-running-training-script-locally)  
- [Interpretability Demo](#-interpretability-demo)  
- [Project Workflow](#-project-workflow)  
- [Requirements](#-requirements)  

---

## Setup

### 1. Create Virtual Environment & Install Dependencies
```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

---

## 🎵 Data Preprocessing

### Step 1: Enter the preprocessing directory
```bash
$ cd data_processing
```

### Step 2: Convert audio files to 16kHz WAV

#### Toxic dataset
```bash
$ python3 wav_converter.py <toxic_dataset_directory_path> <wav_directory_path> fine_tuned_model
```

#### Non-Toxic dataset
```bash
$ python3 wav_converter.py <non_toxic_dataset_directory_path> <wav_directory_path> non-fine_tuned_model
```

### Step 3: Generate CSV with labels
```bash
$ python3 csv_generator.py <data_set_directory> <csv_file_path>
```

This CSV will be used for fine-tuning the model.  

---

## Training the Model

Training is performed in **Google Colab** (recommended for GPU acceleration).  

1. Upload your CSV and WAV dataset to **Google Drive**.  
2. Open the Colab notebook (provided in this repo).  
3. Fine-tune the **Bangla Wav2Vec2.0** model with Hugging Face `transformers`.  
4. Save the trained model back to Drive for later use.

---

## Interpretability Demo

After fine-tuning, you can explain predictions using **Integrated Gradients** and **Saliency Maps**.

### Run Interpretability Script
```bash
$ python3.11 demo_interpretability.py     --model_path "/Users/nishatanjum/aml_project/train-model/toxic/wav2vec2_bangla_toxic_model"     --audio "/Users/nishatanjum/aml_project/train-model/test/uniq_non-toxic.wav"     --out "outputs_example_non_toxic"
```

### Parameters
- `--model_path`: Path to fine-tuned model (local or Drive).  
- `--audio`: Input audio file (16kHz mono `.wav`).  
- `--out`: Output folder where interpretability results will be saved.  

### Outputs
- **attention_rollout** → Predicted label & probabilities  
- **waveform.png** → Original audio waveform  
- **ig_attribution.png** → Integrated Gradients attribution plot  
- **saliency_attribution.png** → Saliency attribution plot  
- **.npy files** → Raw arrays for reproducibility  

---

## Project Workflow

1. **Preprocess data** → Convert raw audio to WAV + generate CSV.  
2. **Fine-tune model** → Train in Colab (or locally).  
3. **Evaluate** → Run predictions on new audio.  
4. **Explain** → Apply interpretability (IG + Saliency).
