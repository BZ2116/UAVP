"""
YOLO 测试集推理脚本（带置信度保存）
"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import YOLO_MODEL_PATH, FINAL_DATA_DIR, ensure_dir

PROJECT_ROOT = Path(__file__).parent.parent


def run_prediction():
    """
    执行 YOLO 推理并保存结果
    """
MODEL_PATH = YOLO_MODEL_PATH
    TEST_IMAGES = FINAL_DATA_DIR / "images" / "test"
    CONF_THRESHOLD = 0.20
    IOU_THRESHOLD = 0.45
    IMG_SIZE = 640
    OUTPUT_DIR = PROJECT_ROOT / "runs" / "predict_lowlight"

    ensure_dir(OUTPUT_DIR)

print("\n" + "=" * 70)
    print("🔍 推理前检查")
    print("=" * 70)

    if not MODEL_PATH.exists():
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("   请先运行 train1.py 训练模型，或确认 YOLO_MODEL_PATH 配置")
        return
    else:
        print(f"✅ 模型文件: {MODEL_PATH}")

    if not TEST_IMAGES.exists():
        print(f"❌ 测试集目录不存在: {TEST_IMAGES}")
        print("   请先运行 Preprocessing/run_pipeline.py 准备数据")
        return
    else:
        test_imgs = list(TEST_IMAGES.glob("*.jpg")) + list(TEST_IMAGES.glob("*.png"))
        print(f"✅ 测试集目录: {TEST_IMAGES}")
        print(f"   测试图像数量: {len(test_imgs)} 张")

    print("=" * 70 + "\n")

print("📦 加载 YOLO 模型...")
    model = YOLO(str(MODEL_PATH))
    print("✅ 模型加载成功！\n")

print("=" * 70)
    print("🚀 开始推理")
    print("=" * 70)
    print(f"   置信度阈值: {CONF_THRESHOLD}")
    print(f"   IOU 阈值: {IOU_THRESHOLD}")
    print(f"   图像尺寸: {IMG_SIZE}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print("=" * 70 + "\n")

    results = model.predict(
        source=str(TEST_IMAGES),
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
        verbose=True,
    )

print("\n" + "=" * 70)
    print("✅ 推理完成！")
    print("=" * 70)

    labels_dir = OUTPUT_DIR / "labels"
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        print(f"📁 结果保存位置:")
        print(f"   标签文件（带 conf）: {labels_dir}")
        print(f"   可视化结果: {OUTPUT_DIR}")
        print(f"\n📊 统计信息:")
        print(f"   生成标签文件数: {len(label_files)} 个")

        if label_files:
            sample_file = label_files[0]
            print(f"\n📄 示例标签文件: {sample_file.name}")
            with open(sample_file, 'r') as f:
                content = f.read().strip()
                if content:
                    lines = content.split('\n')
                    print(f"   检测到 {len(lines)} 个目标")
                    print(f"   格式示例（class x y w h conf）:")
                    for i, line in enumerate(lines[:3], 1):
                        print(f"      {i}. {line}")
                    if len(lines) > 3:
                        print(f"      ... (还有 {len(lines) - 3} 行)")
                else:
                    print(f"   该图像无检测结果（可能无目标或低于阈值）")

        non_empty = sum(1 for f in label_files if os.path.getsize(f) > 0)
        print(f"\n   有检测结果的图像: {non_empty} / {len(label_files)}")
        print(f"   空结果图像: {len(label_files) - non_empty}")

    print("=" * 70)
    print("\n💡 下一步:")
    print(f"   1. 检查结果文件: {labels_dir}/*.txt")
    print("   2. 每个 txt 文件格式: class x y w h conf")
    print("   3. 将 labels_dir 路径配置到 utils/config.py 的 YOLO_LABELS_DIR")
    print("\n🎯 如需调整置信度阈值，请修改 CONF_THRESHOLD 参数")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    run_prediction()
