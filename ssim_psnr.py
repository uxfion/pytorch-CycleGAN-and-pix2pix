import os
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np

# weight_index = 6
weight_names = [
    "0.效果较好_ultrasound_2023_10_10_batch5",
    # "1.nature_ultrasound",
    # "2.no_nature",
    # "3.1_damage",
    # "4.no_hyper",
    # "5.no_perceptual",
    # "6.raw_cyclegan",
    # "7.changsha",
]

blur = 8
clear = 8

for weight_index in [7]:
    # original_dir = './datasets/changsha/changsha_2024_05_14_test_hr'
    # contrast_dir = f'./datasets/changsha/results_sr/changsha_{blur}_{clear}/rand_damage_{blur}_clear_{clear}_weight_{weight_names[weight_index]}'
    original_dir = './results/tim_主观评价/'
    contrast_dir = f'./datasets/changsha/results_sr/changsha_{blur}_{clear}/rand_damage_{blur}_clear_{clear}_weight_{weight_names[weight_index]}'

    ssim_values = []
    psnr_values = []

    # 获取目录下的所有图像文件
    original_files = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f)) and f.endswith(".jpg")]

    # print(f"{len(original_files)} images found in {original_dir}")


    # 遍历所有文件
    for file in original_files:
        index = file.split('-')[0]  # 获取文件的索引
        contrast_file = file

        original_path = os.path.join(original_dir, file)
        contrast_path = os.path.join(contrast_dir, contrast_file)
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
        print(f"{original_path.replace('./datasets/changsha/changsha_2024_05_14_', '')} vs {contrast_path.replace('./datasets/changsha/results_sr/changsha_8_8/', '').replace('')}")

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
    print(f"{blur} blur, {clear} clear")
    print(f"original_dir: {original_dir}")
    print(f"contrast_dir: {contrast_dir}")
    print(f"Weights: {weight_names[weight_index]}")
    print(f"Mean SSIM: {mean_ssim}")
    print(f"Mean PSNR: {mean_psnr} dB")


