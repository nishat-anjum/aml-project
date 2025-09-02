import os
import csv
import argparse
import logging


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

def generate_csv(wav_file_directory, metadata_file_path):
    logger.debug("Start generating CSV for data Set")
    dataset = []
    for filename in os.listdir(wav_file_directory):
        if filename.lower().endswith(".wav"):
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
    parser.add_argument("audio_wav_directory", type=str, help="Root directory containing wav files.")
    parser.add_argument("output_csv_file_path", type=str, help="file to same file path")
    args = parser.parse_args()
    generate_csv(args.audio_wav_directory, args.output_csv_file_path)


