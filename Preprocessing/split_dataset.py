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
    将 NEU-DET 数据集按前缀规则划分为 train / val / test 三个子集。

    功能说明：
        - 从 root/images 中读取全部图片，根据文件名前缀确定类别
        - 每个类别划分为 180 / 60 / 60
        - 将图片及对应 txt 标签复制到 handled_data 下
        - 自动生成 YOLO 所需的 data.yaml 文件

    参数：
        root (Path): NEU-DET 数据集根目录（应包含 images/ 和 labels/）
    """
    root = Path(root)
    img_dir = root / "ll_data"
    label_dir = root / "labels"
    project_root = Path(__file__).parent.parent
    out = project_root / "data" / "final_data"

    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

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

    random.seed(42)
    splits = {"train": [], "val": [], "test": []}

    for cls, imgs in files_by_class.items():
        random.shuffle(imgs)
        splits["train"].extend(imgs[:180])
        splits["val"].extend(imgs[180:240])
        splits["test"].extend(imgs[240:300])

    def copy_files(file_list, split_name):
        for img_path in file_list:
            lbl_path = label_dir / (img_path.stem + ".txt")

            shutil.copy(img_path, out / "images" / split_name / img_path.name)

            if lbl_path.exists():
                shutil.copy(lbl_path, out / "labels" / split_name / lbl_path.name)

    copy_files(splits["train"], "train")
    copy_files(splits["val"], "val")
    copy_files(splits["test"], "test")

    yaml_content = f"""path: {out.absolute()}
train: images/train
val: images/val
test: images/test

nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
"""

    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("划分完成！")
    print(f"训练集: {len(splits['train'])} 张")
    print(f"验证集: {len(splits['val'])} 张")
    print(f"测试集（VLM最终评估）: {len(splits['test'])} 张")
    print(f"data.yaml 已生成 → {out/'data.yaml'}")
    print(f"训练命令示例：")
    print(f"yolo train data={out}/data.yaml model=yolo11n.pt epochs=30 imgsz=640 device=0")
