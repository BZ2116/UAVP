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


class UAVPPromptEngine:
    def __init__(self):
        self.template = UAVP_TEMPLATE

    def get_cus_guidance(self, cus_score: float):
        """
        根据 CUS 得到提示强度，而不是直接判定缺陷严重程度。
        CUS 仅用于描述检测结果的可信程度和提示语气。
        """
        if cus_score > CUS_HIGH_THRESHOLD:
            guidance_level = "High"
            hint = (
                "检测结果较可靠且缺陷区域影响较明显。请重点关注该区域，"
                "但最终严重程度仍需结合图像内容、缺陷类别、形态和位置综合判断。"
            )
        elif cus_score > CUS_MEDIUM_THRESHOLD:
            guidance_level = "Medium"
            hint = (
                "检测结果具有一定参考价值。请结合图像特征进行客观分析，"
                "必要时建议人工复核。"
            )
        else:
            guidance_level = "Low"
            hint = (
                "检测结果可信度或区域影响较弱，可能存在误检或低可见性问题。"
                "请使用谨慎语气，并强调人工复检。"
            )

        return guidance_level, hint

    def get_prompt(self, image_id: str, detections):
        det_lines = []

        for i, det in enumerate(detections):
            area = det['area_ratio']
            conf = det['confidence']

            cus_score = CUS_CONFIDENCE_WEIGHT * conf + CUS_AREA_WEIGHT * area
            guidance_level, hint = self.get_cus_guidance(cus_score)

            line = (
                f"- 目标 {i+1}: 类别={det['defect_type']}, "
                f"位置={det['location']}, "
                f"检测置信度={conf:.2f}, "
                f"面积占比={area:.4f}\n"
                f"  [CUS提示] CUS评分={cus_score:.4f}, 提示强度={guidance_level}\n"
                f"  [分析引导] {hint}"
            )
            det_lines.append(line)

        return self.template.format(
            image_id=image_id,
            count=len(detections),
            detection_list_str="\n".join(det_lines)
        )


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
