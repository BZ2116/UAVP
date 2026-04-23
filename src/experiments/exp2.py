"""
Exp2: 空间坐标提示模式（YOLO bbox 坐标注入，无置信度引导）
"""
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    VLMAnalyzer, clean_and_save_json, YOLOParser,
    EXP2_OUTPUT_DIR, YOLO_LABELS_DIR, VLM_MODEL_PATH,
    FINAL_DATA_DIR, EXP2_TEMPLATE,
)


# =================================================================
# 模块 1: 提示词工程 (空间坐标提示模式 - Exp 2)
# =================================================================
class UAVPPromptEngineExp2:
    def __init__(self):
        self.template = EXP2_TEMPLATE

    def get_prompt(self, image_id: str, detections):
        det_lines = []
        for i, det in enumerate(detections):
            # Exp2: 仅注入坐标和初步类别，无置信度引导（与 Full-UAVP 对比的关键差异）
            line = f"- 目标 {i+1}: 疑似类别={det['defect_type']}, 坐标位置={det['location']}。"
            det_lines.append(line)

        return self.template.format(
            image_id=image_id,
            count=len(detections),
            detection_list_str="\n".join(det_lines) if det_lines else "无"
        )


# =================================================================
# 主流程
# =================================================================
def main():
    IMAGE_DIR = FINAL_DATA_DIR / "images" / "test"
    LABEL_DIR = YOLO_LABELS_DIR
    SAVE_DIR = EXP2_OUTPUT_DIR

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Exp2 使用轻量版解析器（无需 conf），符合实验设计
    parser = YOLOParser().parse_label_file_light
    prompt_engine = UAVPPromptEngineExp2()
    analyzer = VLMAnalyzer(VLM_MODEL_PATH)

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')]
    if not image_files:
        print(f"❌ 测试图像目录为空: {IMAGE_DIR}")
        return

    print(f"\n🚀 Exp 2 (图像+空间提示) 开始，共 {len(image_files)} 张图像...")

    success, failed, skipped = 0, 0, 0
    with tqdm(total=len(image_files), desc="处理进度", unit="img") as pbar:
        for img_file in image_files:
            img_path = IMAGE_DIR / img_file
            txt_path = LABEL_DIR / img_file.rsplit('.', 1)[0] + ".txt"

            detections = parser(str(txt_path))
            if not detections:
                skipped += 1
                pbar.update(1)
                continue

            prompt = prompt_engine.get_prompt(img_file, detections)

            try:
                raw_output = analyzer.analyze(str(img_path), prompt)
                save_path = SAVE_DIR / img_file.rsplit('.', 1)[0] + ".json"
                if clean_and_save_json(raw_output, str(save_path)):
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n❌ 处理出错 {img_file}: {e}")
                failed += 1
            pbar.update(1)

    print(f"\n✅ Exp 2 完成: 成功 {success}, 失败 {failed}, 跳过(无检测框) {skipped}")
    print(f"📁 结果保存: {SAVE_DIR}")


if __name__ == "__main__":
    main()
