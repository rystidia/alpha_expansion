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

_GC_BASE = "https://raw.githubusercontent.com/anmolagarwal999/GrabCut-based-Image-Segmentation/master"

_MB2001 = "https://vision.middlebury.edu/stereo/data/scenes2001/data"
_MB2003 = "https://vision.middlebury.edu/stereo/data/scenes2003/newdata"

MIDDLEBURY_STEREO = {
    "venus": (
        f"{_MB2001}/venus/im2.ppm",
        f"{_MB2001}/venus/im6.ppm",
        f"{_MB2001}/venus/disp2.pgm",
        8,
    ),
    "cones": (
        f"{_MB2003}/cones/im2.png",
        f"{_MB2003}/cones/im6.png",
        f"{_MB2003}/cones/disp2.png",
        4,
    ),
}

def load_middlebury_stereo(name: str):
    """Returns (left, right, gt, disparity_scale) as float32/uint arrays."""
    if name == "tsukuba":
        from ci_data import load_tsukuba
        left, right, gt = load_tsukuba()
        return left, right, gt, 16
    if name not in MIDDLEBURY_STEREO:
        raise KeyError(f"unknown scene: {name}; choices: tsukuba, {list(MIDDLEBURY_STEREO)}")
    left_url, right_url, gt_url, scale = MIDDLEBURY_STEREO[name]
    ext = left_url.rsplit(".", 1)[-1]
    left_path = _download(left_url, os.path.join(ROOT, "data", name, f"left.{ext}"))
    right_path = _download(right_url, os.path.join(ROOT, "data", name, f"right.{ext}"))
    gt_ext = gt_url.rsplit(".", 1)[-1]
    gt_path = _download(gt_url, os.path.join(ROOT, "data", name, f"gt.{gt_ext}"))
    left = np.array(Image.open(left_path)).astype(np.float32)
    right = np.array(Image.open(right_path)).astype(np.float32)
    gt = np.array(Image.open(gt_path))
    return left, right, gt, scale


GRABCUT_IMAGES = {
    "flower": (
        f"{_GC_BASE}/images/flower.jpg",
        f"{_GC_BASE}/ground_truth/flower.bmp",
        (139, 77, 319, 313),
    ),
    "teddy": (
        f"{_GC_BASE}/images/teddy.jpg",
        f"{_GC_BASE}/ground_truth/teddy.bmp",
        (59, 53, 172, 279),
    ),
    "llama": (
        f"{_GC_BASE}/images/llama.jpg",
        f"{_GC_BASE}/ground_truth/llama.bmp",
        (110, 102, 255, 268),
    ),
}


def load_segmentation_image(name: str):
    if name not in GRABCUT_IMAGES:
        raise KeyError(name)
    img_url, mask_url, bbox = GRABCUT_IMAGES[name]
    img_path  = os.path.join(ROOT, "data", "segmentation", f"{name}.jpg")
    mask_path = os.path.join(ROOT, "data", "segmentation", f"{name}_gt.bmp")
    _download(img_url, img_path)
    _download(mask_url, mask_path)
    rgb = np.array(Image.open(img_path).convert("RGB"))
    gt  = np.array(Image.open(mask_path).convert("L"))
    return rgb, gt, bbox



COMMUNITY_GRAPHS = {
    "football": "http://www-personal.umich.edu/~mejn/netdata/football.zip",
}


def load_community_graph(name: str):
    import networkx as nx
    if name == "karate":
        return nx.karate_club_graph()
    if name == "lesmis":
        return nx.les_miserables_graph()
    if name not in COMMUNITY_GRAPHS:
        raise KeyError(f"unknown graph: {name}; choices: karate, lesmis, {list(COMMUNITY_GRAPHS)}")
    import zipfile, io, urllib.request
    url = COMMUNITY_GRAPHS[name]
    cache_path = os.path.join(ROOT, "data", "community", f"{name}.gml")
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as r:
            with zipfile.ZipFile(io.BytesIO(r.read())) as z:
                gml_name = next(n for n in z.namelist() if n.endswith(".gml"))
                with z.open(gml_name) as f:
                    with open(cache_path, "wb") as out:
                        out.write(f.read())
    return nx.read_gml(cache_path, label="id")
