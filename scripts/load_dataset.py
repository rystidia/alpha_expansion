import os
import urllib.request
import numpy as np
from PIL import Image

TSUKUBA_URLS = {
    "left.ppm": "https://vision.middlebury.edu/stereo/data/scenes2001/data/tsukuba/scene1.row3.col3.ppm",
    "right.ppm": "https://vision.middlebury.edu/stereo/data/scenes2001/data/tsukuba/scene1.row3.col4.ppm",
    "gt.pgm": "https://vision.middlebury.edu/stereo/data/scenes2001/data/tsukuba/truedisp.row3.col3.pgm",
}


def download_if_missing(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for filename, url in TSUKUBA_URLS.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)


def load_tsukuba():
    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "tsukuba")
    )

    download_if_missing(data_dir)

    left_img = Image.open(os.path.join(data_dir, "left.ppm"))
    right_img = Image.open(os.path.join(data_dir, "right.ppm"))

    gt_img = Image.open(os.path.join(data_dir, "gt.pgm"))

    left = np.array(left_img).astype(np.float32)
    right = np.array(right_img).astype(np.float32)
    gt = np.array(gt_img)

    return left, right, gt
