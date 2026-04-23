"""
共享常量配置
类别顺序需与 data.yaml 中 names 列表完全一致：
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
"""

# ==================== 类别映射 ====================
CLASS_MAP = {
    0: "Crazing",
    1: "Inclusion",
    2: "Patches",
    3: "Pitted Surface",
    4: "Rolled-in Scale",
    5: "Scratches"
}

# ==================== CUS 评分配置 ====================
# Composite Uncertainty Score = CUS = α * conf + β * area_ratio
CUS_CONFIDENCE_WEIGHT = 0.85
CUS_AREA_WEIGHT = 0.15

# CUS 阈值定义（用于 severity 分级）
CUS_HIGH_THRESHOLD = 0.70
CUS_MEDIUM_THRESHOLD = 0.40

# 纯面积阈值（用于 Mode-D / area-only 消融实验）
AREA_HIGH_THRESHOLD = 0.15
AREA_MEDIUM_THRESHOLD = 0.05
