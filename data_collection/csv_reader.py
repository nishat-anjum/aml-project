import pandas as pd
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

def read_playlist_from_csv_file(file_path):
    try:
        logger.debug("Reading CSV file")
        data = pd.read_csv(file_path)
        return data["playlist_url"].tolist()
    except FileNotFoundError:
        raise Exception(f"CSV file not found at {file_path}")
    except KeyError:
        raise Exception(F"CSV file must have a column named")