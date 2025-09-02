import os
from pydub import AudioSegment
import logging
from config import MP3_AUDIO_DIRECTORY, WAV_FILE_DIRECTORY, AUDIO_SPLIT_DURATION

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

def split_audio():

    if not os.path.exists(MP3_AUDIO_DIRECTORY):
        raise FileNotFoundError(f"The  '{MP3_AUDIO_DIRECTORY}' was not found.")

    if not os.path.exists(WAV_FILE_DIRECTORY):
        os.makedirs(WAV_FILE_DIRECTORY)

    counter = 0
    for filename in os.listdir(MP3_AUDIO_DIRECTORY):
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(MP3_AUDIO_DIRECTORY, filename)
            audio = AudioSegment.from_mp3(mp3_path)
            filtered_audio = audio.set_channels(1).set_frame_rate(16000)
            audio_length = len(filtered_audio)

            #TODO Filter noise


            for start_ms in range(0, audio_length, AUDIO_SPLIT_DURATION):
                end_ms = min(start_ms + AUDIO_SPLIT_DURATION, audio_length)
                chunk = filtered_audio[start_ms:end_ms]
                counter += 1
                output_filename = f"toxic_{counter}.mp3"
                output_path = os.path.join(WAV_FILE_DIRECTORY, output_filename)

                chunk.export(output_path, format="mp3")
                print(f"Exported: {output_path}")
    print("Processing complete!")
