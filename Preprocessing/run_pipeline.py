"""
author:Bruce Zhao
date: 2025/12/5 12:02
"""
# run_pipeline.py
from pathlib import Path
from lowlight import process_dataset_linear
from split_dataset import split_neu_det


if __name__ == "__main__":

    root = Path("../data/NEU-DET")

    # Step 1: Low-light enhancement
    original_images = root / "images"
    low_light_output = root / "ll_data"

    print("\n===== STEP 1: Low-Light Enhancement =====")
    process_dataset_linear(original_images, low_light_output)

    # Step 2: Replace original images with low-light images for split
    print("\n===== STEP 2: Split Dataset (using low-light images) =====")

    # You may need to copy low-light → root/images before splitting
    # or directly modify split_neu_det() to use final_data
    # For now, do this (overwrite images used for splitting):
    target_images = root / "images"
    target_images.mkdir(parents=True, exist_ok=True)

    # Copy low-light images into root/images
    for img_path in low_light_output.iterdir():
        if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            dest = target_images / img_path.name
            dest.write_bytes(img_path.read_bytes())

    split_neu_det(root)
