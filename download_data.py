from urllib.request import urlretrieve
from dotenv import load_dotenv, find_dotenv
from zipfile import ZipFile
import os
import progressbar
import urllib

# create a progress report bar
pbar = None

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


if __name__ == "__main__":
    dotenv_path = find_dotenv(".env")
    load_dotenv(dotenv_path)

    # dataset url: https://universe.roboflow.com/yolov7test-u13vc/weapon-detection-m7qso
    # change the url variable to the custom made download url for your account
    # example: url = "https://universe.roboflow.com/ds/abcdefg?key=123"
    url = os.getenv("DATASET_URL")
    filename = "remapped-train-80-val-20_coco.zip"

    # download the file
    path, _ = urlretrieve(url, filename, show_progress)

    # unzip the file
    with ZipFile(path, 'r') as zip_object:
        zip_object.extractall(path="./remapped-train-80-val-20_coco")
