"""
项目路径配置
所有路径统一从项目根目录计算，避免硬编码绝对路径
"""
import os
from pathlib import Path

# 项目根目录：UAVP/
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# ==================== 数据目录 ====================
DATA_DIR = PROJECT_ROOT / "data"
NEU_DET_DIR = DATA_DIR / "NEU-DET"
FINAL_DATA_DIR = DATA_DIR / "final_data"
LL_DATA_DIR = NEU_DET_DIR / "ll_data"

# ==================== YOLO 推理结果目录 ====================
# YOLO 低光检测标签输出目录（由 yolo_predict.py 生成）
YOLO_LABELS_DIR = PROJECT_ROOT / "res_final" / "detect_res" / "lowlight" / "labels"

# ==================== VLM 实验输出目录 ====================
EXP1_OUTPUT_DIR = PROJECT_ROOT / "compare1"
EXP2_OUTPUT_DIR = PROJECT_ROOT / "compare22"
UAVP_OUTPUT_DIR = PROJECT_ROOT / "uavp_out"

# ==================== 消融实验输出目录 ====================
ABLATION_A_DIR = PROJECT_ROOT / "compare22"
ABLATION_B_DIR = PROJECT_ROOT / "ablation_resultsB"
ABLATION_C_DIR = PROJECT_ROOT / "ablation_resultsDD"
ABLATION_D_DIR = PROJECT_ROOT / "res_final" / "json_output" / "Exp3"

# ==================== 评估基准与输出 ====================
GT_JSON_DIR = PROJECT_ROOT / "res_final" / "gt_json"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "evaluation_lowlight"

# ==================== 模型配置 ====================
VLM_MODEL_PATH = os.environ.get(
    "VLM_MODEL_PATH",
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

YOLO_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "neu_det_results" / "yolo11n_normal_ultra" / "weights" / "best.pt"

# ==================== 辅助函数 ====================
def ensure_dir(path: Path) -> Path:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path
