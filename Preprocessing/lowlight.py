"""
低光化处理模块

严格按论文公式实现：
- 公式 (3.1): I_low = I_norm^γ * k + α * (I - G_σ * I)
- 公式 (3.2): 自适应 Gamma 因子
- 公式 (3.3): 叠加 AWGN 噪声后 clip

author: Bruce Zhao
"""
import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


class LowLightProcessor:
    """
    低光化处理器。
    严格遵循论文公式，依次执行：自适应 Gamma 亮度变换 → 高频补偿 → 噪声叠加。
    """

    GAMMA_RANGE = (1.5, 2.0)
    K_RANGE = (0.75, 0.9)
    ALPHA = 0.25
    SIGMA_GAUSSIAN = 2
    NOISE_RANGE = (2, 4)

    def __init__(self,
                 gamma_range=None,
                 k_range=None,
                 alpha=None,
                 sigma_gaussian=None,
                 noise_range=None):
        self.gamma_range = gamma_range or self.GAMMA_RANGE
        self.k_range = k_range or self.K_RANGE
        self.alpha = alpha if alpha is not None else self.ALPHA
        self.sigma_gaussian = sigma_gaussian if sigma_gaussian is not None else self.SIGMA_GAUSSIAN
        self.noise_range = noise_range or self.NOISE_RANGE

    def transform(self, img: np.ndarray) -> np.ndarray:
        """
        按公式 (3.1) ~ (3.3) 对单张图像执行低光化处理。

        Args:
            img: BGR 图像，uint8

        Returns:
            低光化后的 BGR 图像，uint8
        """
        I_norm = img.astype(np.float32) / 255.0

        avg_brightness = np.mean(I_norm)
        gamma_base = random.uniform(*self.gamma_range)
        if avg_brightness > 0.6:
            gamma = gamma_base + 0.2
        else:
            gamma = gamma_base

        k = random.uniform(*self.k_range)
        I_pow = np.power(I_norm, gamma)
        I_brightness = I_pow * k

        I_smoothed = gaussian_filter(I_norm, sigma=self.sigma_gaussian)
        I_high_freq = I_norm - I_smoothed
        I_compensated = I_brightness + self.alpha * I_high_freq

        sigma_n = random.uniform(*self.noise_range)
        noise = np.random.normal(0, sigma_n / 255.0, img.shape)
        I_final = I_compensated + noise

        return np.clip(I_final * 255, 0, 255).astype(np.uint8)

    def process_image(self, img: np.ndarray) -> np.ndarray:
        """对外暴露的接口，与 transform 等价"""
        return self.transform(img)


def lowlight_dataset(input_dir: str,
                     output_dir: str,
                     gamma_range=None,
                     k_range=None,
                     alpha=None,
                     sigma_gaussian=None,
                     noise_range=None,
                     preview: bool = False):
    """
    对整个数据集执行低光化处理。

    Args:
        input_dir:  输入图像目录路径
        output_dir: 输出图像目录路径
        gamma_range: Gamma 因子范围，默认 (1.5, 2.0)
        k_range:     亮度缩放系数范围，默认 (0.75, 0.9)
        alpha:       高频补偿系数，默认 0.25
        sigma_gaussian: 高斯滤波器标准差，默认 2
        noise_range: 噪声强度 σ_n 范围，默认 (2, 4)
        preview:     是否预览前 3 张对比图，默认 False

    Returns:
        None
    """
    processor = LowLightProcessor(
        gamma_range=gamma_range,
        k_range=k_range,
        alpha=alpha,
        sigma_gaussian=sigma_gaussian,
        noise_range=noise_range,
    )

    os.makedirs(output_dir, exist_ok=True)

    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    if not files:
        print(f"❌ 错误：{input_dir} 中没有找到图像文件！")
        return

    print(f"\n🌙 低光化处理开始，共 {len(files)} 张图像")
    print(f"   Gamma 范围: {gamma_range or (1.5, 2.0)}")
    print(f"   亮度系数 k: {k_range or (0.75, 0.9)}")
    print(f"   高频补偿 α: {alpha if alpha is not None else 0.25}")
    print(f"   高斯标准差 σ: {sigma_gaussian if sigma_gaussian is not None else 2}")
    print(f"   噪声强度 σ_n: {noise_range or (2, 4)}")

    for filename in tqdm(files, desc="低光化处理"):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        img = cv2.imread(src_path)
        if img is None:
            print(f"\n⚠️ 无法读取: {filename}")
            continue

        img_low = processor.transform(img)
        cv2.imwrite(dst_path, img_low)

    print(f"\n✅ 处理完成，{len(files)} 张图像已保存至: {output_dir}")


if __name__ == '__main__':
    lowlight_dataset(
        input_dir=r"E:\.1Study\code\PY\UAVP\data\NEU-DET\images",
        output_dir=r"E:\.1Study\code\PY\UAVP\data\NEU-DET\ll_data"
    )
