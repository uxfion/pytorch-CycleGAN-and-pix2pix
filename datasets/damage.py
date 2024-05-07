from PIL import Image, ImageFilter
import random
import os
from tqdm import tqdm


def simulate_checkerboarding(image, scale_factor):
    # Reduce the size of the image by the scale factor
    small_image = image.resize((image.size[0]//scale_factor, image.size[1]//scale_factor), Image.NEAREST)
    # Scale it back up to the original size
    result_image = small_image.resize(image.size, Image.NEAREST)
    return result_image


def random_effect(image, n):
    choice = random.randint(1, 4)
    # choice = 1
    if choice == 1:
        return simulate_checkerboarding(image, n)
    elif choice == 2:
        return image.filter(ImageFilter.GaussianBlur(radius=n))
    elif choice == 3:
        image = simulate_checkerboarding(image, n)
        return image.filter(ImageFilter.GaussianBlur(radius=n))
    elif choice == 4:
        image = image.filter(ImageFilter.GaussianBlur(radius=n))
        return simulate_checkerboarding(image, n)
    else:
        return image

if __name__ == "__main__":
    # image = Image.open("7-LR.jpg")
    # image = random_effect(image, 6)
    # image.show()
    # image.save("damaged_image.jpg")


    # random effect on low-reslution directory
    # 源文件夹路径
    source_dir = './xijing/high_quality/'
    # 目标文件夹路径
    target_dir = './xijing/hr_rand_damage_8/'

    # 图片后缀列表（如果有其他格式也可以加入）
    img_exts = ['.jpg', '.png', '.jpeg', '.bmp', '.gif']

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in tqdm(os.listdir(source_dir)):
    # 检查文件是否为图片
        if os.path.splitext(filename)[1] in img_exts:
            image = Image.open(os.path.join(source_dir, filename))
            image = random_effect(image, 8)
            image.save(os.path.join(target_dir, filename))
