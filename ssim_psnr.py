import os
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np

blur = 8
clear = 2

original_dir = './datasets/all/test'
print(f"original_dir: {original_dir}")
contrast_dir = f'./results/xijing/9restormer/test_degradation{blur}_sr{clear}'
print(f"contrast_dir: {contrast_dir}")

ssim_values = []
psnr_values = []

# 获取目录下的所有图像文件
original_files = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f)) and f.endswith(".jpg")]

# sort
original_files.sort()

print(f"{len(original_files)} images found in {original_dir}")


# 遍历所有文件
for file in original_files:

    original_path = os.path.join(original_dir, file)
    contrast_path = os.path.join(contrast_dir, file)
    # 确保对比图像存在
    if not os.path.exists(contrast_path):
        print(f"Contrast image not found for {file}")
        continue

    # 读取图像
    original = Image.open(original_path)
    contrast = Image.open(contrast_path)

    if original.size != contrast.size:
        # print(f"{file} original.size: {original.size}")
        # print(f"{contrast_file} contrast.size: {contrast.size}")
        # print("Resizing images to equal size")
        contrast = contrast.resize(original.size)

    original = np.array(original)
    contrast = np.array(contrast)

    # print(f"{file} vs {contrast_file}")
    print(f"{original_path} vs {contrast_path}")

    # 计算SSIM
    ssim_value = compare_ssim(original, contrast, multichannel=True, win_size=3)
    ssim_values.append(ssim_value)
    print(f"SSIM: {ssim_value}")

    # 计算PSNR
    psnr_value = compare_psnr(original, contrast)
    psnr_values.append(psnr_value)
    print(f"PSNR: {psnr_value} dB")
    print("\n")

mean_ssim = np.mean(ssim_values)
mean_psnr = np.mean(psnr_values)

mean_ssim = round(mean_ssim, 3)
mean_psnr = round(mean_psnr, 3)

print("==============================")
print(f"degradation {blur}, clear {clear}")
print(f"original_dir: {original_dir}")
print(f"contrast_dir: {contrast_dir}")
print(f"Mean SSIM: {mean_ssim}")
print(f"Mean PSNR: {mean_psnr} dB")
