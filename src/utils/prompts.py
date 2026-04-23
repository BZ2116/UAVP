"""
UAVP Prompt 模板集中管理
"""


# =================================================================
# Exp1: 纯图像盲测
# =================================================================
EXP1_TEMPLATE = """
你是一位资深的工业视觉质检专家。

请观察该低光照钢材表面图像，检测其中明显的工业缺陷。

常见缺陷类型包括：
- Crazing（裂纹）
- Inclusion（夹杂）
- Patches（斑块）
- Pitted Surface（点蚀）
- Rolled-in_scale（氧化皮压入）
- Scratches（划痕）

### 分析要求：
1. 最多检测 3 个最明显的缺陷
2. 只报告你有较高信心的缺陷
3. 优先关注图像中心区域及纹理异常区域
4. 位置格式为 [x_center, y_center, width, height]（归一化 0~1）

### 输出要求：
必须输出 JSON：

{{
  "image_id": "{image_id}",
  "lighting_condition": "low_light",
  "defects": [
    {{
      "defect_type": "类别名称",
      "location": [x, y, w, h],
      "confidence_score": 0.00,
      "severity": "High/Medium/Low",
      "visual_analysis": "简要描述",
      "suggestion": "维修建议"
    }}
  ],
  "overall_assessment": {{
    "defect_count": "整数",
    "risk_level": "High/Medium/Low"
  }}
}}
"""


# =================================================================
# Exp2: 空间坐标提示
# =================================================================
EXP2_TEMPLATE = """
你是一位资深的工业视觉质检专家。
请根据以下前置模型提供的潜在缺陷坐标，对图像进行重点目标分析。

### 图像信息
- 图像 ID: {image_id}
- 光照条件: 低光照生产环境

### 潜在缺陷区域列表 (共计: {count} 个目标)
{detection_list_str}

### UAVP 分析指令:
检测器在上述坐标区域发现了疑似缺陷。请你仔细观察这些指定区域：
1. 判断该区域是否为真实缺陷？（剔除背景干扰或正常纹理）
2. 确认或修正该缺陷的类别，并给出你的视觉分析理由。

### 输出规范:
你必须输出一个且仅一个合法的 JSON 对象，格式如下。严禁包含任何多余的对话文本。
{{
  "image_id": "{image_id}",
  "lighting_condition": "low_light",
  "defects": [
    {{
      "defect_type": "缺陷类别名称",
      "location": [x, y, w, h],
      "confidence_score": 1.00,
      "severity": "High/Medium/Low",
      "visual_analysis": "对该给定区域视觉特征的详细描述",
      "suggestion": "专业的维修与工艺建议"
    }}
  ],
  "overall_assessment": {{
    "defect_count": {count},
    "risk_level": "整体风险等级 (High/Medium/Low)"
  }}
}}
"""


# =================================================================
# Full-UAVP: CUS 复合评分引导
# =================================================================
UAVP_TEMPLATE = """
你是一位资深的工业视觉质检专家。
请根据以下 CV 模型提供的检测列表，对提供的图像进行深度多目标分析。

### 图像信息
- 图像 ID: {image_id}
- 光照条件: 低光照生产环境

### 检测列表 (共计: {count} 个目标)
{detection_list_str}

### UAVP分析指令:
系统已基于 CUS (Composite Uncertainty Score) 提供评估与引导语，请严格结合视觉特征进行最终判断：

- High等级: 使用确定性语气，直接给出修复优先级
- Medium等级: 客观描述并建议人工复核
- Low等级: 使用"疑似"语气并强调人工复检

### 输出规范:
你必须输出一个且仅一个合法的 JSON 对象，格式如下。严禁包含任何多余文本。

{{
  "image_id": "{image_id}",
  "lighting_condition": "low_light",
  "defects": [
    {{
      "defect_type": "缺陷类别名称",
      "location": [x, y, w, h],
      "confidence_score": 0.00,
      "severity": "High/Medium/Low",
      "visual_analysis": "视觉分析",
      "suggestion": "维修建议"
    }}
  ],
  "overall_assessment": {{
    "defect_count": {count},
    "risk_level": "High/Medium/Low"
  }}
}}
"""


# =================================================================
# Ablation 实验通用模板
# =================================================================
ABLATION_TEMPLATE = """
你是一位资深的工业视觉质检专家。
请对以下检测到的缺陷进行深度视觉分析。

### 图像信息
- 图像 ID: {image_id}
- 光照条件: 低光照生产环境

### 检测列表 (共计: {count} 个目标)
{detection_list_str}

### 输出规范:
你必须输出一个且仅一个合法的 JSON 对象，格式如下。严禁包含任何多余文字。
{{
  "image_id": "{image_id}",
  "lighting_condition": "low_light",
  "defects": [
    {{
      "defect_type": "缺陷类别名称",
      "location": [x, y, w, h],
      "confidence_score": 0.00,
      "severity": "High/Medium/Low",
      "visual_analysis": "视觉分析",
      "suggestion": "维修建议"
    }}
  ],
  "overall_assessment": {{
    "defect_count": {count},
    "risk_level": "High/Medium/Low"
  }}
}}
"""
