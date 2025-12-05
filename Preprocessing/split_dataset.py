"""
author: Bruce Zhao
date: 2025/12/4 14:28
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict


def split_neu_det(root):
    """
    Split NEU-DET into train/val/test datasets by image prefix.

    Args:
        root (Path): Path of NEU-DET dataset (should contain 'images' and 'labels')

    Result:
        Creates 'handled_data' directory with:
            images/train, images/val, images/test
            labels/train, labels/val, labels/test
        And generates data.yaml
    """

    root = Path(root)
    img_dir = root / "images"
    label_dir = root / "labels"

    out = root / "final_data"

    # Create target directory structure
    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Collect images by class using filename prefix
    files_by_class = defaultdict(list)
    for img_path in img_dir.glob("*.jpg"):
        prefix = img_path.stem[:2].upper()

        if   prefix in ["CR", "CZ"]: cls = 0  # crazing
        elif prefix in ["IN", "IC"]: cls = 1  # inclusion
        elif prefix in ["PA", "PT"]: cls = 2  # patches
        elif prefix in ["PS", "PI"]: cls = 3  # pitted_surface
        elif prefix in ["RS", "RO"]: cls = 4  # rolled-in_scale
        elif prefix in ["SC", "SR"]: cls = 5  # scratches
        else:
            print(f"未知前缀: {img_path.stem}, 已跳过")
            continue

        files_by_class[cls].append(img_path)

    # 2. Split 180 / 60 / 60 per class
    random.seed(42)
    splits = {"train": [], "val": [], "test": []}

    for cls, imgs in files_by_class.items():
        random.shuffle(imgs)
        splits["train"].extend(imgs[:180])
        splits["val"].extend(imgs[180:240])
        splits["test"].extend(imgs[240:300])

    # 3. Copy files
    def copy_files(file_list, split_name):
        for img_path in file_list:
            lbl_path = label_dir / (img_path.stem + ".txt")

            shutil.copy(img_path, out / "images" / split_name / img_path.name)

            if lbl_path.exists():
                shutil.copy(lbl_path, out / "labels" / split_name / lbl_path.name)

    copy_files(splits["train"], "train")
    copy_files(splits["val"], "val")
    copy_files(splits["test"], "test")

    # 4. Create data.yaml
    yaml_content = f"""path: {out.absolute()}
train: images/train
val: images/val
test: images/test

nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
"""

    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # Summary
    print("划分完成！")
    print(f"训练集: {len(splits['train'])} 张")
    print(f"验证集: {len(splits['val'])} 张")
    print(f"测试集（VLM最终评估）: {len(splits['test'])} 张")
    print(f"data.yaml 已生成 → {out/'data.yaml'}")
    print(f"训练命令示例：")
    print(f"yolo train data={out}/data.yaml model=yolo11n.pt epochs=30 imgsz=640 device=0")
