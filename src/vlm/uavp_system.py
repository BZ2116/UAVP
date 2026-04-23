"""
Full-UAVP 系统：基于 CUS 复合评分的 VLM 深度分析
"""
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    VLMAnalyzer, clean_and_save_json, YOLOParser,
    VLM_MODEL_PATH, FINAL_DATA_DIR, YOLO_LABELS_DIR, UAVP_OUTPUT_DIR,
    CLASS_MAP,
    CUS_CONFIDENCE_WEIGHT, CUS_AREA_WEIGHT,
    CUS_HIGH_THRESHOLD, CUS_MEDIUM_THRESHOLD,
    UAVP_TEMPLATE,
)


# =================================================================
# UAVP Prompt Engine
# =================================================================
class UAVPPromptEngine:
    def __init__(self):
        self.template = UAVP_TEMPLATE

    def get_prompt(self, image_id: str, detections):
        det_lines = []
        for i, det in enumerate(detections):
            area = det['area_ratio']
            conf = det['confidence']

            # CUS = α * conf + β * area_ratio
            cus_score = CUS_CONFIDENCE_WEIGHT * conf + CUS_AREA_WEIGHT * area

            if cus_score > CUS_HIGH_THRESHOLD:
                severity = "High"
                hint = "特征显著，请使用确定性语气，直接判定为高风险并优先处理。"
            elif cus_score > CUS_MEDIUM_THRESHOLD:
                severity = "Medium"
                hint = "特征较明显，请客观描述并建议人工复核。"
            else:
                severity = "Low"
                hint = "特征微弱，可能误检，请使用'疑似'语气并强调人工复检。"

            line = (
                f"- 目标 {i+1}: 类别={det['defect_type']}, "
                f"位置={det['location']}, "
                f"置信度={conf:.2f}, "
                f"面积占比={area:.4f}\n"
                f"  [系统评估] CUS评分={cus_score:.4f}, 严重程度={severity}\n"
                f"  [引导语] {hint}"
            )
            det_lines.append(line)

        return self.template.format(
            image_id=image_id,
            count=len(detections),
            detection_list_str="\n".join(det_lines)
        )


# =================================================================
# 主流程
# =================================================================
def main():
    IMAGE_DIR = FINAL_DATA_DIR / "images" / "test"
    LABEL_DIR = YOLO_LABELS_DIR
    SAVE_DIR = UAVP_OUTPUT_DIR

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    parser = YOLOParser(CLASS_MAP)
    prompt_engine = UAVPPromptEngine()
    analyzer = VLMAnalyzer(VLM_MODEL_PATH)

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')]
    if not image_files:
        print(f"❌ 测试图像目录为空: {IMAGE_DIR}")
        return

    print(f"\n🚀 Full-UAVP 开始，共 {len(image_files)} 张图像...")

    success, failed, skipped = 0, 0, 0
    for img_file in tqdm(image_files, desc="UAVP Processing"):
        img_path = IMAGE_DIR / img_file
        txt_path = LABEL_DIR / img_file.replace(".jpg", ".txt")

        detections = parser.parse_label_file(str(txt_path))
        if not detections:
            skipped += 1
            continue

        prompt = prompt_engine.get_prompt(img_file, detections)

        try:
            raw_output = analyzer.analyze(str(img_path), prompt)
            save_path = SAVE_DIR / img_file.replace(".jpg", ".json")
            if clean_and_save_json(raw_output, str(save_path)):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Error processing {img_file}: {e}")
            failed += 1

    print(f"\n✅ UAVP 完成: 成功 {success}, 失败 {failed}, 跳过(无检测框) {skipped}")
    print(f"📁 结果保存: {SAVE_DIR}")


if __name__ == "__main__":
    main()
