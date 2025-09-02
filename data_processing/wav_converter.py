import os
import uuid
import csv
import argparse
import logging
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def configure_log(module_name):
    logger = logging.getLogger(module_name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

logger = configure_log(__name__)

def resample_audio_pydub(filename,
                         audio_file_path,
                         data_set_label,
                         wav_file_directory,
                         target_sr=16000,
                         silence_threshold=-40,
                         min_silence_length=800):
    try:
        audio = AudioSegment.from_file(audio_file_path)
        nonSilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=min_silence_length,
            silence_thresh=silence_threshold,
        )

        if not nonSilent_ranges:
            print(f"Ignoring {filename}: Contains only silence.")
            return

        audio = audio.set_frame_rate(target_sr).set_channels(1)
        file_id = uuid.uuid4().hex
        output_filename = f"{file_id}_{data_set_label}.wav"
        output_path = os.path.join(wav_file_directory, output_filename)
        audio.export(output_path, format="wav")

    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")


def create_dataset(mp3_audio_directory, output_wav_audio_directory, data_type):
    if not os.path.exists(output_wav_audio_directory):
        os.makedirs(output_wav_audio_directory)
    if mp3_audio_directory and os.path.exists(mp3_audio_directory):
        for root, _, files in os.walk(mp3_audio_directory):
            for filename in files:
                logger.debug(f'convert {data_type}-{filename} to wav')
                mp3_path = os.path.abspath(os.path.join(root, filename))
                logger.debug(mp3_path)
                resample_audio_pydub(filename, mp3_path, data_type, output_wav_audio_directory)

    print("Wav Conversion completed!")


def generate_csv(wav_file_directory, metadata_file_path):
    logger.debug("Start generating CSV for data Set")
    dataset = []
    for filename in os.listdir(wav_file_directory):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(wav_file_directory, filename)
            audio = AudioSegment.from_wav(file_path)
            samples = np.array(audio.get_array_of_samples())
            labelString = filename.split('_')[1].split('.')[0]
            label = 0 if "non-fine_tuned_model" in filename.lower() else 1
            dataset.append({"file": filename, "label": labelString})

    with open(metadata_file_path, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['file', 'label']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    print("Csv Generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio files to wav and creat dataset")
    parser.add_argument("audio_directory", type=str, help="Root directory containing MP3 files.")
    parser.add_argument("output_wav_directory", type=str, help="Root directory to store wav files.")
    parser.add_argument("data_type", type=str, help="data label")
    args = parser.parse_args()
    create_dataset(args.audio_directory, args.output_wav_directory, args.data_type)


