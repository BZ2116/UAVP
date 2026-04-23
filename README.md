# UAVP: Uncertainty-Aware Visual Prompting

> 低光照钢材表面缺陷多模态分析 | 本科毕业设计 2025.12 - 2026.05

---

## 系统架构

<img width="886" height="586" alt="image" src="https://github.com/user-attachments/assets/8fa418b6-5fcf-411e-9c90-821e2e73e31d" />


---

## 项目简介

提出 **UAVP（Uncertainty-Aware Visual Prompting）** 机制，将 YOLO 检测的置信度与缺陷面积融合为 CUS 复合评分，通过语义引导词控制 Qwen2.5-VL-7B-Instruct 的推理语气，实现低光照条件下从缺陷图像到结构化 JSON 报告的端到端生成。

**数据集**：NEU-DET 钢材表面缺陷（6类：Crazing、Inclusion、Patches、Pitted Surface、Rolled-in Scale、Scratches）

---

## 核心算法

### CUS 复合评分

```
CUS = 0.85 × Confidence + 0.15 × Area Ratio
```

| CUS 区间      | 风险等级 | 语气          |
| ----------- | ------ | ------------- |
| > 0.70      | High   | 确定性语气，优先处理  |
| 0.40 ~ 0.70 | Medium | 客观描述，建议人工复核 |
| ≤ 0.40      | Low    | "疑似"语气，强调复检   |

---

## 项目结构

```
UAVP/
├── Preprocessing/               # 数据预处理
│   ├── lowlight.py             # 低光化（论文公式实现）
│   ├── split_dataset.py        # 数据划分
│   └── run_pipeline.py        # 一键流水线
├── src/
│   ├── utils/                  # 公共模块
│   ├── yolo/                  # YOLO 训练与推理
│   ├── vlm/                   # VLM 分析系统
│   └── experiments/           # 实验脚本
└── data/NEU-DET/             # 原始数据
```


---

## 快速开始

```bash
# 1. 数据预处理
python Preprocessing/run_pipeline.py

# 2. 训练 YOLO
python src/yolo/yolo_train.py

# 3. YOLO 推理
python src/yolo/yolo_predict.py

# 4. VLM 实验
python src/experiments/exp1.py          # 纯图像盲测
python src/experiments/exp2.py          # 空间坐标提示
python src/vlm/uavp_system.py           # Full-UAVP
python src/experiments/ablation.py --mode A   # 消融实验

# 5. 评估
python src/experiments/evaluate.py
```

---

## 实验结果

![对比实验结果图]() <!-- 待添加：准确率对比图 -->

---

## 环境配置

```bash
pip install -r requirements.txt
export VLM_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"   # 可选
```

所有路径在 `src/utils/config.py` 中统一管理。

---

## 作者

- **赵耀** | 重庆邮电大学 计算机科学与技术学院
- **指导教师**：刘俊
- 感谢龙雪同学在标注与实验讨论中的贡献
