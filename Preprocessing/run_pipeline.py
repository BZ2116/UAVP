"""
author:Bruce Zhao
date: 2025/12/5 12:02
"""
# run_pipeline.py
from pathlib import Path
from lowlight import process_dataset_linear
from split_dataset import split_neu_det


if __name__ == "__main__":
    # 数据目录
    root = Path("../data/NEU-DET")
    original_images = root / "images"
    low_light_output = root / "ll_data"

    print("\n===== STEP 1: 对原始图像进行低光处理 =====")
    process_dataset_linear(original_images, low_light_output)

    print("\n===== STEP 2: 划分数据集  =====")

    target_images = root / "images"
    target_images.mkdir(parents=True, exist_ok=True)

    # 将低光后的图像覆盖原 images/ 中的图片
    for img_path in low_light_output.iterdir():
        if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            dest = target_images / img_path.name
            dest.write_bytes(img_path.read_bytes())
    # 按NEU-DET 规则进行划分（train/val/test）
    split_neu_det(root)
