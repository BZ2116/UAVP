"""
Exp1: 纯图像盲测基线（无 YOLO 坐标注入）
"""
import os
import sys
from pathlib import Path
from tqdm import tqdm

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import VLMAnalyzer, clean_and_save_json, EXP1_OUTPUT_DIR, VLM_MODEL_PATH, EXP1_TEMPLATE


# =================================================================
# 模块 1: Prompt 工程（纯图像输入模式 - Exp 1）
# =================================================================
class UAVPPromptEngineExp1:
    def __init__(self):
        self.template = EXP1_TEMPLATE

    def get_prompt(self, image_id: str) -> str:
        return self.template.format(image_id=image_id)


# =================================================================
# 主流程
# =================================================================
def main():
    # 使用 utils.config 中的路径（从环境变量或相对路径读取）
    from utils.config import FINAL_DATA_DIR

    IMAGE_DIR = FINAL_DATA_DIR / "images" / "test"
    SAVE_DIR = EXP1_OUTPUT_DIR

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    analyzer = VLMAnalyzer(VLM_MODEL_PATH)
    prompt_engine = UAVPPromptEngineExp1()

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')]
    if not image_files:
        print(f"❌ 测试图像目录为空: {IMAGE_DIR}")
        return

    print(f"\n🚀 Exp 1 (纯图像盲测) 开始，共 {len(image_files)} 张图像...")

    success, failed = 0, 0
    with tqdm(total=len(image_files), desc="处理进度", unit="img") as pbar:
        for img_file in image_files:
            img_path = IMAGE_DIR / img_file
            prompt = prompt_engine.get_prompt(img_file)

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

    print(f"\n✅ Exp 1 完成: 成功 {success}, 失败 {failed}")
    print(f"📁 结果保存: {SAVE_DIR}")


if __name__ == "__main__":
    main()
