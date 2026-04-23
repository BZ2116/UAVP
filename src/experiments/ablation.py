"""
消融实验脚本（支持 A/B/C/D 四种方案）
用法: python ablation.py --mode A
或设置环境变量: ABLATION_MODE=A python ablation.py
"""
import sys
import os
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    VLMAnalyzer, clean_and_save_json, YOLOParser,
    VLM_MODEL_PATH, FINAL_DATA_DIR, YOLO_LABELS_DIR,
    ABLATION_A_DIR, ABLATION_B_DIR, ABLATION_C_DIR, ABLATION_D_DIR,
    CUS_CONFIDENCE_WEIGHT, CUS_AREA_WEIGHT,
    CUS_HIGH_THRESHOLD, CUS_MEDIUM_THRESHOLD,
    AREA_HIGH_THRESHOLD, AREA_MEDIUM_THRESHOLD,
    ABLATION_TEMPLATE,
)


# =================================================================
# 消融实验 PromptEngine（支持4组方案）
# =================================================================
class AblationPromptEngine:
    """
    消融实验方案说明：
    - Mode A: 仅 Bbox（无置信度、无面积）—— 基线
    - Mode B: Bbox + 置信度（hard threshold 0.5）
    - Mode C: Bbox + CUS（0.85*conf + 0.15*area）—— 与 Full-UAVP 一致
    - Mode D: 仅面积（无置信度）—— 验证面积独立作用
    """

    def __init__(self, mode="C"):
        self.mode = mode.upper()

    def get_prompt(self, image_id: str, detections):
        det_lines = []
        for i, det in enumerate(detections):
            area = det['area_ratio']
            conf = det['confidence']

            if self.mode == "A":
                line = f"- 目标 {i+1}: 类别={det['defect_type']}, 位置={det['location']}"

            elif self.mode == "B":
                hint = "特征明确，请直接判定为高风险并给出修复建议。" if conf > 0.5 else "特征微弱，可能为误检，请以'疑似'语气描述并建议人工复检。"
                line = (f"- 目标 {i+1}: 类别={det['defect_type']}, 位置={det['location']}, 置信度={conf:.2f}\n"
                        f"  [引导语] {hint}")

            elif self.mode == "C":
                # CUS = 0.85 * conf + 0.15 * area，与 README 定义一致
                cus = CUS_CONFIDENCE_WEIGHT * conf + CUS_AREA_WEIGHT * area
                if cus > CUS_HIGH_THRESHOLD:
                    hint = "特征显著，请直接判定为高风险并优先处理。"
                elif cus > CUS_MEDIUM_THRESHOLD:
                    hint = "特征较明显，请客观描述并建议人工复核。"
                else:
                    hint = "特征微弱，可能误检，请以'疑似'语气描述并强调人工复检。"
                line = (f"- 目标 {i+1}: 类别={det['defect_type']}, 位置={det['location']}, "
                        f"置信度={conf:.2f}, 面积占比={area:.4f}, CUS={cus:.3f}\n"
                        f"  [引导语] {hint}")

            elif self.mode == "D":
                # 纯面积驱动，阈值与 uavp_system.py 的 infer_severity 对齐
                if area > AREA_HIGH_THRESHOLD:
                    hint = "缺陷面积较大，建议判定为高风险并优先处理。"
                elif area > AREA_MEDIUM_THRESHOLD:
                    hint = "缺陷面积中等，请客观描述并建议人工复核。"
                else:
                    hint = "缺陷面积较小，可能为误检，请以'疑似'语气描述并建议人工复检。"
                line = (f"- 目标 {i+1}: 类别={det['defect_type']}, 位置={det['location']}, "
                        f"面积占比={area:.4f}\n"
                        f"  [引导语] {hint}")

            else:
                raise ValueError(f"未知 Mode: {self.mode}，可选 A/B/C/D")

            det_lines.append(line)

        return ABLATION_TEMPLATE.format(
            image_id=image_id,
            count=len(detections),
            detection_list_str="\n".join(det_lines)
        )


# =================================================================
# 辅助：根据 mode 获取输出目录
# =================================================================
ABLATION_DIR_MAP = {
    "A": ABLATION_A_DIR,
    "B": ABLATION_B_DIR,
    "C": ABLATION_C_DIR,
    "D": ABLATION_D_DIR,
}


# =================================================================
# 主流程
# =================================================================
def main():
    parser_arg = argparse.ArgumentParser(description="UAVP 消融实验")
    parser_arg.add_argument(
        "--mode", "-m",
        default=os.environ.get("ABLATION_MODE", "C"),
        choices=["A", "B", "C", "D"],
        help="消融实验模式 (默认从环境变量 ABLATION_MODE 读取)"
    )
    args = parser_arg.parse_args()
    MODE = args.mode.upper()

    IMAGE_DIR = FINAL_DATA_DIR / "images" / "test"
    LABEL_DIR = YOLO_LABELS_DIR
    SAVE_DIR = ABLATION_DIR_MAP[MODE]

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    parser = YOLOParser()
    prompt_engine = AblationPromptEngine(mode=MODE)
    analyzer = VLMAnalyzer(VLM_MODEL_PATH)

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')]
    if not image_files:
        print(f"❌ 测试图像目录为空: {IMAGE_DIR}")
        return

    print(f"\n🚀 消融实验 Mode-{MODE} 开始，共 {len(image_files)} 张图像...")

    success, failed, skipped = 0, 0, 0
    with tqdm(total=len(image_files), desc=f"Mode-{MODE}", unit="img") as pbar:
        for img_file in image_files:
            img_path = IMAGE_DIR / img_file
            txt_path = LABEL_DIR / img_file.rsplit('.', 1)[0] + ".txt"

            detections = parser.parse_label_file(str(txt_path))
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

    print(f"\n✅ Mode-{MODE} 完成: 成功 {success}, 失败 {failed}, 跳过(无检测框) {skipped}")
    print(f"📁 结果保存: {SAVE_DIR}")


if __name__ == "__main__":
    main()
