import os
import urllib.request
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _download(url: str, target_path: str):
    if os.path.exists(target_path):
        return target_path
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"downloading {url}")
    try:
        urllib.request.urlretrieve(url, target_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {url} -> {target_path}.\n"
            f"Download manually with:\n  curl -L -o {target_path} {url}\n"
            f"Error: {e}"
        )
    return target_path


RESTORATION_IMAGES = {
    "airplane": "https://sipi.usc.edu/database/preview/misc/5.1.11.png",
    "house":    "https://sipi.usc.edu/database/preview/misc/4.1.05.png",
    "peppers":  "https://sipi.usc.edu/database/preview/misc/4.2.07.png",
}


def load_restoration_image(name: str) -> np.ndarray:
    """Return an HxW grayscale uint8 image."""
    if name not in RESTORATION_IMAGES:
        raise KeyError(f"unknown image: {name}; choices: {list(RESTORATION_IMAGES)}")
    path = os.path.join(ROOT, "data", "restoration", f"{name}.png")
    _download(RESTORATION_IMAGES[name], path)
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def add_noise(image: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)
