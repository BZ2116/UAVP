
from pathlib import Path
import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# --- 配置参数 (线性模型) ---
BRIGHTNESS_MIN_FACTOR = 0.3  # 最暗为原始亮度的 30%
BRIGHTNESS_MAX_FACTOR = 0.6  # 最亮为原始亮度的 60%
NOISE_STD = 10               # 高斯噪声标准差，模拟传感器噪声



def create_low_light_image_linear(image):
    """
    通过线性缩减亮度和添加高斯噪声来模拟低光照。
    Args:
        image (np.array): BGR 格式的原始图像。
    Returns:
        np.array: 模拟低光照后的图像。
    """

    # 1. 随机选择亮度缩减因子
    factor = np.random.uniform(BRIGHTNESS_MIN_FACTOR, BRIGHTNESS_MAX_FACTOR)

    # 2. 线性缩减亮度：I' = I * factor
    # 这会均匀地降低所有像素值，保留相对对比度。
    darkened_image = image.astype(np.float32) * factor

    # 3. 添加高斯噪声 (模拟低光传感器噪声)
    noise = np.random.normal(0, NOISE_STD, image.shape).astype(np.float32)

    # 4. 叠加并确保像素值在 0-255 范围内
    final_image = darkened_image + noise
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    return final_image


def process_dataset_linear(input_dir, output_dir):
    """
    遍历原始数据集，生成并保存低光图像。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"检测到 {len(image_files)} 张图像，开始生成线性低光版本...")

    for filename in tqdm(image_files):
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = cv2.imread(img_path)
        if img is None:
            continue

        low_light_img = create_low_light_image_linear(img)

        cv2.imwrite(output_path, low_light_img)

    print("\n线性低光环境合成完成！")
