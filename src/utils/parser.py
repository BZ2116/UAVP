"""
共享的 YOLO 标签解析器
"""
import os
from pathlib import Path
from .constants import CLASS_MAP


class YOLOParser:
    """
    解析 YOLO 检测结果 txt 文件。
    YOLO txt 格式：class x_center y_center width height confidence
    """

    def __init__(self, class_map=None):
        self.class_map = class_map or CLASS_MAP

    def parse_label_file(self, txt_path: str):
        results = []
        if not os.path.exists(txt_path):
            return results

        with open(txt_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                if len(data) == 6:
                    cls_id, x, y, w, h, conf = data
                    area_ratio = float(w) * float(h)
                    results.append({
                        "defect_type": self.class_map.get(int(cls_id), "Unknown"),
                        "location": [float(x), float(y), float(w), float(h)],
                        "confidence": float(conf),
                        "area_ratio": round(area_ratio, 4)
                    })
        return results

    def parse_label_file_light(self, txt_path: str):
        """
        轻量版解析（无置信度），用于 exp2 等只需要坐标和类别的场景
        """
        results = []
        if not os.path.exists(txt_path):
            return results

        with open(txt_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                if len(data) >= 5:
                    cls_id, x, y, w, h = data[:5]
                    results.append({
                        "defect_type": self.class_map.get(int(cls_id), "Unknown"),
                        "location": [float(x), float(y), float(w), float(h)]
                    })
        return results
