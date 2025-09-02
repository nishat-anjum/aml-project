import os
import logging
from pytube import YouTube, Playlist
from config import MP3_AUDIO_DIRECTORY

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

def download_audio_from_youtube_video(url):
    try:
        yt = YouTube(url)
        logger.debug(f"downloading: {yt.title}")
        audio_stream = yt.streams.filter(only_audio=True).first()
        if not os.path.exists(MP3_AUDIO_DIRECTORY):
            os.mkdir(MP3_AUDIO_DIRECTORY)
        output_file = audio_stream.download(output_path=MP3_AUDIO_DIRECTORY)
        base, ext = os.path.splitext(output_file)
        new_file = base + ".mp3"
        os.rename(output_file, new_file)

    except Exception as e:
        logger.error("Error downloading audio")
        raise e


def download_audio_from_playlist(playlist_url):
    try:
        playlist = Playlist(playlist_url)
        logger.debug(f"Downloading audio from playlist: {playlist.title}")
        for video_url in playlist.video_urls:
            try:
                download_audio_from_youtube_video(video_url)
                print(f"Downloaded: {playlist.title}")
            except Exception as e:
                print(f"Failed to download playlist: {e}")
    except Exception as e:
        print(f"Error processing playlist: {e}")
