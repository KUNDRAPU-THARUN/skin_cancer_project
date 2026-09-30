import argparse
from pathlib import Path

from organize_ham10000 import organize_ham10000
from preprocess import process_dataset
from train import main as train_model


def run_pipeline(metadata_csv: str, image_dir: str, raw_output: str = "data/raw", processed_dir: str = "data/processed"):
    raw_root = Path(raw_output)
    raw_root.mkdir(parents=True, exist_ok=True)

    if Path(metadata_csv).exists() and Path(image_dir).exists():
        organize_ham10000(metadata_csv, image_dir, raw_output)

    process_dataset(input_dir=raw_output, output_dir=processed_dir)
    train_model(dataset_dir=processed_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full skin-cancer training pipeline.")
    parser.add_argument("--metadata", default="", help="Path to HAM10000 metadata.csv")
    parser.add_argument("--images", default="", help="Path to image folder containing HAM10000 images")
    parser.add_argument("--raw-output", default="data/raw", help="Folder where class directories will be created")
    parser.add_argument("--processed-dir", default="data/processed", help="Folder for processed images")
    args = parser.parse_args()

    if not args.metadata or not args.images:
        print("No dataset provided. To run the full pipeline, pass both --metadata and --images.")
        print("Example: python src/pipeline.py --metadata data/HAM10000_metadata.csv --images data/HAM10000_images")
    else:
        run_pipeline(args.metadata, args.images, args.raw_output, args.processed_dir)
