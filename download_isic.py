# download_isic.py
import os
import argparse
import requests
from pathlib import Path
from time import sleep
from urllib.parse import urlencode

API_BASE = "https://api.isic-archive.com/api/v1"

def fetch_assets(query_params, max_items=500):
    """Return list of image metadata dicts matching query_params."""
    results = []
    page = 0
    per_page = 100
    while len(results) < max_items:
        params = dict(query_params)
        params.update({"offset": page*per_page, "limit": per_page})
        url = f"{API_BASE}/image?{urlencode(params)}"
        res = requests.get(url, timeout=30)
        if res.status_code != 200:
            print("ISIC API error", res.status_code, res.text)
            break
        data = res.json()
        if not data:
            break
        results.extend(data)
        if len(data) < per_page:
            break
        page += 1
        sleep(0.2)
    return results[:max_items]

def download_image(image_id, save_path):
    url = f"{API_BASE}/image/{image_id}/download"
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(1024*10):
                f.write(chunk)
        return True
    else:
        print(f"Failed to download {image_id} -> {r.status_code}")
        return False

def prepare_folders(out_dir):
    (out_dir / "train" / "benign").mkdir(parents=True, exist_ok=True)
    (out_dir / "train" / "malignant").mkdir(parents=True, exist_ok=True)
    (out_dir / "val" / "benign").mkdir(parents=True, exist_ok=True)
    (out_dir / "val" / "malignant").mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="output folder")
    parser.add_argument("--num-per-class", type=int, default=500, help="how many images per class")
    parser.add_argument("--val-split", type=float, default=0.2, help="fraction to reserve for val")
    args = parser.parse_args()

    out_dir = Path(args.out)
    prepare_folders(out_dir)

    # Query ISIC images with benign_malignant=benign and malignant
    for label in ["benign", "malignant"]:
        print("Fetching metadata for:", label)
        meta_list = fetch_assets({"has_images": "true", "benign_malignant": label}, max_items=args.num_per_class*2)
        # filter for dermoscopic images if possible (but we keep broad)
        # shuffle
        import random
        random.shuffle(meta_list)
        selected = meta_list[: args.num_per_class]
        print(f"Selected {len(selected)} items for {label}")
        # split into train/val
        n_val = int(len(selected) * args.val_split)
        train = selected[n_val:]
        val = selected[:n_val]
        for subset, items in [("train", train), ("val", val)]:
            folder = out_dir / subset / label
            for item in items:
                img_id = item.get("_id") or item.get("id") or item.get("image_name")
                # determine filename
                fname = f"{img_id}.jpg"
                save_path = folder / fname
                if save_path.exists():
                    continue
                ok = download_image(img_id, str(save_path))
                if not ok:
                    continue
                # quick check file size
                if save_path.stat().st_size < 1000:
                    print("Tiny file, removing", save_path)
                    save_path.unlink(missing_ok=True)
                sleep(0.1)
        print(f"Downloaded {label} images to {out_dir}")

    print("Done. Check", out_dir)

if __name__ == "__main__":
    main()
