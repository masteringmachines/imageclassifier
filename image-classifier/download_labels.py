"""
download_labels.py
Downloads ImageNet class labels from PyTorch Hub and saves them
as model/imagenet_classes.json (a list of 1000 class names).

Run once before starting the app:
    python download_labels.py
"""

import json
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
OUT = Path(__file__).parent / "model" / "imagenet_classes.json"

def download():
    print(f"Downloading ImageNet class labels from PyTorch Hub…")
    with urllib.request.urlopen(URL) as resp:
        lines = resp.read().decode("utf-8").strip().splitlines()
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(lines, f, indent=2)
    print(f"✅ Saved {len(lines)} classes to {OUT}")

if __name__ == "__main__":
    download()
