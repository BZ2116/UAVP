"""
预处理自动化流水线
依次执行：低光化处理 → 数据集划分 + 生成 data.yaml
"""
from pathlib import Path
from lowlight import lowlight_dataset
from split_dataset import split_neu_det


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    neu_det_dir = project_root / "data" / "NEU-DET"

    original_images = neu_det_dir / "images"
    low_light_output = neu_det_dir / "ll_data"

    print("\n===== STEP 1: 低光化处理 =====")
    lowlight_dataset(
        str(original_images),
        str(low_light_output)
    )

    print("\n===== STEP 2: 划分数据集（train/val/test）并生成 data.yaml =====")
    split_neu_det(str(neu_det_dir))
