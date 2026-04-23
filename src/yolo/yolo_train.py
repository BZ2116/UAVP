"""
YOLOv11n 训练脚本
author: Bruce Zhao
date:   2025/12/11
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import FINAL_DATA_DIR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(FINAL_DATA_DIR / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        project="runs",
        name="yolov11n_neu_best",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        seed=0,
        patience=30,
        cache="disk",
        amp=True,

        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,

        box=7.5,
        cls=0.5,
        dfl=1.5,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        close_mosaic=10,
    )

    print("训练完成！最佳模型:", results.best)
    test_metrics = model.val(data=str(FINAL_DATA_DIR / "data.yaml"), split="test")
    print(f"测试集 mAP@0.5:0.95 = {test_metrics.box.map:.4f}")
