import argparse
import shutil
from pathlib import Path

import pandas as pd

LABEL_MAP = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def find_image_files(image_dir: Path, image_id: str):
    for suffix in (".jpg", ".jpeg", ".png"):
        matches = list(image_dir.rglob(f"{image_id}{suffix}"))
        if matches:
            return matches[0]
    return None


def organize_ham10000(metadata_csv: str, image_dir: str, output_dir: str):
    metadata_path = Path(metadata_csv)
    image_root = Path(image_dir)
    output_root = Path(output_dir)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    df = pd.read_csv(metadata_path)
    required_cols = {"image_id", "dx"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {sorted(missing)}")

    copied = 0
    for row in df.itertuples(index=False):
        image_id = str(row.image_id)
        dx = str(row.dx).strip().lower()
        label_name = LABEL_MAP.get(dx, dx)

        source_path = find_image_files(image_root, image_id)
        if source_path is None:
            continue

        class_dir = output_root / label_name
        class_dir.mkdir(parents=True, exist_ok=True)

        destination = class_dir / source_path.name
        if not destination.exists():
            shutil.copy2(source_path, destination)
        copied += 1

    print(f"Organized {copied} images into {output_root}")
    return output_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize the HAM10000 dataset into class directories.")
    parser.add_argument("--metadata", required=True, help="Path to HAM10000 metadata.csv")
    parser.add_argument("--images", required=True, help="Path to the folder containing HAM10000 images")
    parser.add_argument("--output", default="data/raw", help="Output directory for organized class folders")
    args = parser.parse_args()

    organize_ham10000(args.metadata, args.images, args.output)
