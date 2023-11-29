from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np

# 读取图像
original = Image.open('./results/xijing/pre_exp/low_quality/1-LR.jpg')
contrast = Image.open('./results/xijing/pre_exp/fake/1.png')

if original.size != contrast.size:
    print(f"original.size: {original.size}")
    print(f"contrast.size: {contrast.size}")
    print("Resizing images to equal size")
    contrast = contrast.resize(original.size)

original = np.array(original)
contrast = np.array(contrast)

# 计算SSIM
ssim_value = compare_ssim(original, contrast, multichannel=True, win_size=3)
print(f"SSIM: {ssim_value}")

# 计算PSNR
psnr_value = compare_psnr(original, contrast)
print(f"PSNR: {psnr_value} dB")



