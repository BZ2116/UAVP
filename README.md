# UAVP: Uncertainty-Aware Visual Prompting

> 低光照钢材表面缺陷多模态分析 | 本科毕业设计 2025.12 - 2026.05

---

## 系统架构
<div align="center">
<img width="885" height="493" alt="image" src="https://github.com/user-attachments/assets/bb1a5811-da84-4799-8bbd-ca00777a4117" />
</div>


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


<div align="center">

| CUS 区间      | 风险等级   | 语气          |
| ----------- | ------ | ----------- |
| > 0.70      | High   | 确定性语气，优先处理  |
| 0.40 ~ 0.70 | Medium | 客观描述，建议人工复核 |
| ≤ 0.40      | Low    | "疑似"语气，强调复检 |

</div>


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
<div align="center">
  
 <img width="1200" height="600" alt="accuracy_metrics_comparison" src="https://github.com/user-attachments/assets/00f479b8-bc84-4b88-b22f-789a128b7857" />
 
</div>

---

## 环境配置

```bash
pip install -r requirements.t
```
---

## 致谢
本项目为重庆邮电大学本科毕业设计研究成果。在此诚挚感谢以下个人与机构的支持：

* 指导教师：衷心感谢刘俊老师。从选题、思路梳理到实验设计及论文撰写，刘老师全程给予了悉心指导与宝贵建议，为 UAVP 机制的实现提供了关键支持。

* 团队贡献：感谢龙雪同学在人工 GT Json 标注工作中的辛勤付出，以及在实验讨论中提供的帮助。

* 科研环境：感谢计算机科学与技术学院（示范性软件学院）提供的科研环境与实验室场地。

* 致敬开源：感谢 NEU-DET 数据集的提供者以及相关开源模型的开发者，这些开放资源为本研究奠定了坚实基础。

最后，感谢重庆邮电大学四年来的培养，此地见证了我的学术启蒙与成长。
