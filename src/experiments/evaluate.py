"""
评估脚本：对比主实验（Exp1/Exp2/Exp3）和消融实验（A/B/C/D）的准确率和 BERTScore
"""
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    EXP1_OUTPUT_DIR, EXP2_OUTPUT_DIR, UAVP_OUTPUT_DIR,
    ABLATION_A_DIR, ABLATION_B_DIR, ABLATION_C_DIR, ABLATION_D_DIR,
    GT_JSON_DIR, EVAL_OUTPUT_DIR,
    ensure_dir,
)

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bert_score import score

# ==================== 1. 配置路径 ====================
GT_DIR = GT_JSON_DIR
SAVE_REPORT_DIR = ensure_dir(EVAL_OUTPUT_DIR)

# 主对比实验路径
REPO_DIRS = {
    "Baseline (Exp1)": EXP1_OUTPUT_DIR,
    "Spatial (Exp2)": EXP2_OUTPUT_DIR,
    "Full-UAVP (Exp3)": UAVP_OUTPUT_DIR,
}

# 消融实验路径
ABLATION_DIRS = {
    "Ablation-A (BBox only)": ABLATION_A_DIR,
    "Ablation-B (Hard Threshold)": ABLATION_B_DIR,
    "Ablation-C (Area only)": ABLATION_C_DIR,
    "Ablation-D (Full CUS)": ABLATION_D_DIR,
}

# 类别规范化映射（处理大小写、下划线、横杠差异）
CLASS_MAPPING = {
    "crazing": "crazing",
    "inclusion": "inclusion",
    "patches": "patches",
    "pitted surface": "pitted surface", "pitted_surface": "pitted surface",
    "rolled-in scale": "rolled-in scale", "rolled_in_scale": "rolled-in scale",
    "scratches": "scratches"
}


# ==================== 2. 工具函数 ====================
def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ JSON 加载失败 {path}: {e}")
        return None


def calculate_iou(box1, box2, img_size=200):
    if not box1 or not box2:
        return 0
    # 归一化处理
    if any(v > 1.0 for v in box1):
        box1 = [v / img_size for v in box1]
    if any(v > 1.0 for v in box2):
        box2 = [v / img_size for v in box2]

    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area
    return inter_area / (union_area + 1e-6)


def normalize_val(val):
    if not val:
        return ""
    return str(val).lower().strip().replace("_", " ")


# ==================== 3. 核心评估函数 ====================
def evaluate_group(repo_path, group_name):
    accuracies = {"defect_type": [], "severity": [], "combined": []}
    text_refs, text_cands = [], []
    stats = {"gt_load_fail": 0, "pred_load_fail": 0, "both_loaded": 0}

    gt_files = list(Path(GT_DIR).glob("*.json"))
    if not gt_files:
        print(f"⚠️ GT 目录为空: {GT_DIR}")
        return None

    for gt_file in tqdm(gt_files, desc=f"评估 {group_name}", leave=False):
        gt_data = load_json(gt_file)
        if not gt_data:
            stats["gt_load_fail"] += 1
            continue

        pred_path = Path(repo_path) / gt_file.name
        pred_data = load_json(pred_path)
        if not pred_data:
            stats["pred_load_fail"] += 1
            continue

        stats["both_loaded"] += 1
        gt_list = gt_data.get("defects", [])
        pred_list = pred_data.get("defects", [])

        matched_indices = set()
        for pred in pred_list:
            best_iou = 0
            best_gt_idx = -1
            pred_loc = pred.get("location")

            for idx, gt in enumerate(gt_list):
                if idx in matched_indices:
                    continue
                iou = calculate_iou(pred_loc, gt.get("location"))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_gt_idx != -1 and best_iou > 0.1:
                matched_indices.add(best_gt_idx)
                target_gt = gt_list[best_gt_idx]

                gt_type = CLASS_MAPPING.get(
                    normalize_val(target_gt.get("defect_type", "")),
                    normalize_val(target_gt.get("defect_type", ""))
                )
                pred_type = CLASS_MAPPING.get(
                    normalize_val(pred.get("defect_type", "")),
                    normalize_val(pred.get("defect_type", ""))
                )

                gt_sev = normalize_val(target_gt.get("severity", ""))
                pred_sev = normalize_val(pred.get("severity", ""))

                type_correct = 1 if gt_type == pred_type else 0
                sev_correct = 1 if gt_sev == pred_sev else 0

                accuracies["defect_type"].append(type_correct)
                accuracies["severity"].append(sev_correct)
                accuracies["combined"].append(1 if (type_correct and sev_correct) else 0)

                gt_text = target_gt.get("visibility", "") + " " + target_gt.get("suggestion", "")
                pred_text = pred.get("visual_analysis", pred.get("visibility", "")) + " " + pred.get("suggestion", "")

                text_refs.append(gt_text)
                text_cands.append(pred_text)

        # 漏检惩罚
        unmatched = len(gt_list) - len(matched_indices)
        for _ in range(unmatched):
            accuracies["defect_type"].append(0)
            accuracies["severity"].append(0)
            accuracies["combined"].append(0)

    # BERTScore 计算（增加空列表保护）
    avg_f1 = 0.0
    if text_cands:
        try:
            P, R, F1 = score(text_cands, text_refs, lang="zh", verbose=False)
            avg_f1 = F1.mean().item()
        except Exception as e:
            print(f"\n⚠️ BERTScore 计算失败: {e}")

    result = {
        "Experiment": group_name,
        "Defect_Type_Acc": np.mean(accuracies["defect_type"]) if accuracies["defect_type"] else 0.0,
        "Severity_Acc": np.mean(accuracies["severity"]) if accuracies["severity"] else 0.0,
        "Combined_Acc": np.mean(accuracies["combined"]) if accuracies["combined"] else 0.0,
        "BERTScore_F1": avg_f1,
        "GT_加载失败": stats["gt_load_fail"],
        "预测文件缺失": stats["pred_load_fail"],
        "成功配对": stats["both_loaded"],
    }

    print(f"  → {group_name}: 成功配对 {stats['both_loaded']}, "
          f"GT加载失败 {stats['gt_load_fail']}, 预测缺失 {stats['pred_load_fail']}")

    return result


# ==================== 4. 主评估流程 ====================
def evaluate_all():
    all_metrics = []

    print("=== 主对比实验评估 ===")
    for exp_name, repo_path in REPO_DIRS.items():
        res = evaluate_group(repo_path, exp_name)
        if res:
            all_metrics.append(res)

    print("\n=== 消融实验评估 ===")
    for exp_name, repo_path in ABLATION_DIRS.items():
        res = evaluate_group(repo_path, exp_name)
        if res:
            all_metrics.append(res)

    if not all_metrics:
        print("❌ 没有可评估的结果，请检查路径配置")
        return None

    return pd.DataFrame(all_metrics)


# ==================== 5. 绘图与报告输出 ====================
def plot_results(df):
    if df is None or df.empty:
        print("❌ 无数据可绘图")
        return

    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']

    # 图1：准确率柱状图
    plt.figure(figsize=(14, 7))
    plot_data = df.melt(id_vars="Experiment", value_vars=["Defect_Type_Acc", "Severity_Acc", "Combined_Acc"])
    ax = sns.barplot(data=plot_data, x="variable", y="value", hue="Experiment", palette="viridis")
    plt.axhline(y=0.75, color='r', linestyle='--', label='目标阈值 (75%)')
    plt.title("准确率指标对比（主实验 vs 消融实验）", fontsize=16)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(SAVE_REPORT_DIR / "accuracy_comparison_all.png", dpi=300)

    # 图2：BERTScore 箱线图
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Experiment", y="BERTScore_F1", palette="Set2")
    plt.title("BERTScore_F1 分布对比", fontsize=16)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(SAVE_REPORT_DIR / "bertscore_distribution_all.png", dpi=300)

    # 保存表格（包含指标列和统计列）
    df_table = df.round(4)
    df_table.to_csv(SAVE_REPORT_DIR / "final_evaluation_report.csv", index=False)

    print("\n" + "=" * 80)
    print("📊 最终评估报告汇总（主实验 + 消融实验）:")
    print(df_table.to_string(index=False))
    print("=" * 80)
    print(f"图表与报告已保存至: {SAVE_REPORT_DIR}")


if __name__ == "__main__":
    results_df = evaluate_all()
    plot_results(results_df)
