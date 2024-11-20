import torch
from torchvision import transforms
from models import create_model
from options.test_options import TestOptions
import numpy as np
from PIL import Image

import os
from tqdm import tqdm


def load_cyclegan_model():
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # opt.dataroot = "./datasets/xijing/low_quality"
    opt.dataroot = "./dataset/all/test"
    # opt.name = "ultrasound_2023_10_10_batch5"
    opt.name = "6.raw_cyclegan"
    # opt.name = "12.9trans"
    opt.gpu_ids = [1]
    opt.model = "my_test"
    opt.no_dropout = True
    opt.preprocess = "none"

    # Load the model
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    return model


# Function to perform inference
def cyclegan_infer(model, image_raw, sigma):
    # image_raw = crop_to_divisible_by_four(image_raw)
    # Apply necessary transformations
    transform = transforms.Compose([
        # transforms.Lambda(lambda img: __make_power_2(img, base=4)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image_raw).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        target_code = torch.zeros((1, 20))
        target_code[0, sigma] = 1
        target_code = target_code.to("cuda:1")
        fake_image = model.netG(image.to("cuda:1"), target_code)
        # fake_image = model.netG(image.to("cuda:1"))

    # Convert to PIL image and return
    fake_image = (fake_image.cpu().squeeze(0) + 1) / 2  # Denormalize
    fake_image = transforms.ToPILImage()(fake_image)

    contrast_image = mapped(image_raw, fake_image)

    return contrast_image


# 调整第二张图像img2的亮度和对比度，使其与第一张图像img1相似。
def mapped(img1, img2):
    # 将 img1 和 img2 分解为三个通道
    img1_r, img1_g, img1_b = img1.split()
    img2_r, img2_g, img2_b = img2.split()

    # 分别对这三个通道进行处理
    img2_r = mapped_single_channel(img1_r, img2_r)
    img2_g = mapped_single_channel(img1_g, img2_g)
    img2_b = mapped_single_channel(img1_b, img2_b)

    # 将处理后的三个通道合并为一张彩色图像
    img2 = Image.merge("RGB", (img2_r, img2_g, img2_b))

    return img2


def mapped_single_channel(img1, img2):
    img1_pixels = np.sort(np.array(img1).flatten())
    img2_pixels = np.sort(np.array(img2).flatten())

    img1_low = float(img1_pixels[int(len(img1_pixels) * 0.05)])
    img1_high = float(img1_pixels[int(len(img1_pixels) * 0.95)])
    img2_low = float(img2_pixels[int(len(img2_pixels) * 0.05)])
    img2_high = float(img2_pixels[int(len(img2_pixels) * 0.95)])

    img2_array = np.array(img2, dtype=float)
    scale_factor = ((img2_array - img2_low) / (img2_high - img2_low)) * (img1_high - img1_low) + img1_low
    scale_factor = np.clip(scale_factor, 0, 255, out=scale_factor)

    return Image.fromarray(scale_factor.astype(np.uint8))


if __name__ == '__main__':
    # model = load_cyclegan_model()
    # image_raw = Image.open("./datasets/xijing/low_quality/1-LR.jpg")
    # image_output = cyclegan_infer(model, image_raw, 0)
    # image_output.save("./datasets/xijing/fake/1-LR-0.jpg")

    model = load_cyclegan_model()
    blur = 8
    clear = 2
    clear_list = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    # input_folder = f'./datasets/xijing/Gaussian_{blur}/high_quality_Gaussian_{blur}'  # 输入文件夹路径
    input_folder = './datasets/all/test_degradation8/'  # 输入文件夹路径
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # 获取图片文件列表
    # for clear in clear_list:
        # print(f"\nclear: {clear}")
    output_folder = f'./results/xijing/raw/test_degradation8_sr'  # 输出文件夹路径
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in tqdm(files, desc="Processing Images"):  # 使用tqdm显示进度
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        image_raw = Image.open(input_path).convert("RGB")
        image_output = cyclegan_infer(model, image_raw, clear)
        image_output.save(output_path)
