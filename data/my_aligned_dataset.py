import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFilter, ImageEnhance
import random
import numpy as np

from scipy import ndimage

class ImageDegradation:
    def __init__(self):
        pass
    
    def random_effect(self, image, n):
        """
        随机选择和应用图像降质效果
        image: PIL图像对象
        n: 降质程度参数 (1-3)
        """
        # 将n映射到降质参数
        params = self._get_params_from_n(n)
        
        # 获取所有可能的效果
        effects = [
            # 降采样效果
            lambda: self.downsample_bicubic(image, params['resize_ratio']),
            lambda: self.downsample_bilinear(image, params['resize_ratio']),
            lambda: self.simulate_checkerboarding(image, int(1/params['resize_ratio'])),
            
            # 噪声效果
            lambda: self.add_gaussian_noise(image, params['noise_factor']),
            lambda: self.add_speckle_noise(image, params['speckle_factor']),
            lambda: self.add_salt_pepper_noise(image, params['noise_factor']),
            
            # 模糊效果
            lambda: image.filter(ImageFilter.GaussianBlur(radius=params['gaussian_blur_radius'])),
            lambda: self.apply_motion_blur(image, params['motion_blur_size']),
            
            # 对比度和亮度调整
            lambda: self.adjust_contrast(image, params['contrast_factor']),
            lambda: self.adjust_brightness(image, params['brightness_factor'])
        ]
        
        # 随机选择一种效果
        effect = random.choice(effects)
        return effect()

    def _get_params_from_n(self, n):
        """根据n值(1-3)返回相应的参数"""
        params = {
            1: {  # 轻度降质
                'resize_ratio': 0.75,
                'gaussian_blur_radius': 1.5,
                'motion_blur_size': 3,
                'noise_factor': 0.1,
                'speckle_factor': 0.15,
                'contrast_factor': 0.85,
                'brightness_factor': 0.9,
            },
            2: {  # 中度降质
                'resize_ratio': 0.5,
                'gaussian_blur_radius': 2.5,
                'motion_blur_size': 5,
                'noise_factor': 0.2,
                'speckle_factor': 0.25,
                'contrast_factor': 0.7,
                'brightness_factor': 0.8,
            },
            3: {  # 重度降质
                'resize_ratio': 0.25,
                'gaussian_blur_radius': 3.5,
                'motion_blur_size': 7,
                'noise_factor': 0.3,
                'speckle_factor': 0.35,
                'contrast_factor': 0.5,
                'brightness_factor': 0.7,
            }
        }
        return params.get(n, params[2])  # 默认返回中度降质参数

    # 以下是各种效果的具体实现方法
    def downsample_bicubic(self, image, ratio):
        """使用双三次插值进行降采样"""
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        small = image.resize(new_size, Image.BICUBIC)
        return small.resize(image.size, Image.BICUBIC)

    def downsample_bilinear(self, image, ratio):
        """使用双线性插值进行降采样"""
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        small = image.resize(new_size, Image.BILINEAR)
        return small.resize(image.size, Image.BILINEAR)

    def simulate_checkerboarding(self, image, scale_factor):
        """模拟棋盘格效应"""
        small = image.resize((image.size[0]//scale_factor, image.size[1]//scale_factor), Image.NEAREST)
        return small.resize(image.size, Image.NEAREST)

    def add_gaussian_noise(self, image, factor):
        """添加高斯噪声"""
        img_array = np.array(image)
        noise = np.random.normal(0, factor * 255, img_array.shape)
        noisy_img = img_array + noise
        return Image.fromarray(np.uint8(np.clip(noisy_img, 0, 255)))

    def add_speckle_noise(self, image, factor):
        """添加散粒噪声"""
        img_array = np.array(image)
        noise = np.random.normal(0, factor, img_array.shape)
        noisy_img = img_array + img_array * noise
        return Image.fromarray(np.uint8(np.clip(noisy_img, 0, 255)))

    def add_salt_pepper_noise(self, image, factor):
        """添加椒盐噪声"""
        img_array = np.array(image)
        salt = np.random.random(img_array.shape) < factor/2
        pepper = np.random.random(img_array.shape) < factor/2
        img_array[salt] = 255
        img_array[pepper] = 0
        return Image.fromarray(np.uint8(img_array))

    def apply_motion_blur(self, image, size):
        """应用运动模糊
        image: PIL Image对象
        size: 模糊核大小
        """
        img_array = np.array(image)
        
        # 检查图像维度
        if len(img_array.shape) == 3:
            # 对于RGB图像，分别处理每个通道
            result = np.zeros_like(img_array)
            for channel in range(img_array.shape[2]):
                kernel = np.zeros((size, size))
                kernel[int((size-1)/2), :] = np.ones(size)
                kernel = kernel / size
                result[:,:,channel] = ndimage.convolve(
                    img_array[:,:,channel], 
                    kernel, 
                    mode='reflect'
                )
        else:
            # 对于灰度图像
            kernel = np.zeros((size, size))
            kernel[int((size-1)/2), :] = np.ones(size)
            kernel = kernel / size
            result = ndimage.convolve(img_array, kernel, mode='reflect')
        
        return Image.fromarray(np.uint8(np.clip(result, 0, 255)))

    def adjust_contrast(self, image, factor):
        """调整对比度"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def adjust_brightness(self, image, factor):
        """调整亮度"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

class MyAlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # n_diff = random.randint(1, 10)
        # n_A = random.randint(n_diff + 1, 20)
        # n_B = n_A - n_diff

        n_A = random.randint(1, 3)
        n_B = 0

        # n_range = [0, 19]
        # n_A = random.randint(min(n_range[0], n_range[1]), max(n_range[0], n_range[1]))
        # n_B = random.randint(min(n_range[0], n_range[1]), n_A)
        # # check if the path contains 'ultrasound'
        # if 'ultrasound' in AB_path:
        #     n_B = 0  # not blurring B for ultrasound images
        # else:
        #     # if the image is an ultrasound image, we only blur image A and not image B
        #     # if the image is not an ultrasound image, we blur both image A and image B
        #     n_B = random.randint(min(n_range[0], n_range[1]), n_A)

        # A = AB.filter(ImageFilter.GaussianBlur(radius=n_A))
        # B = AB.filter(ImageFilter.GaussianBlur(radius=n_B))

        degrader = ImageDegradation()
        A = degrader.random_effect(AB, n_A)
        B = AB
        # B = self.simulate_checkerboarding(AB, n_B)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        A_blur = n_A
        # print('A_blur:', A_blur)
        B_blur = n_B
        # print('B_blur:', B_blur)
        # blur = float(A_blur - B_blur)/50.0

        blur = A_blur - B_blur

        # print('sub:', blur)
        # A_blur = float(AB_path.split('/')[-1].split('_')[-1].split('.')[0])/100.0

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'blur': blur}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def random_effect(self, image, n):
        choice = random.randint(1, 4)
        # choice = 2
        if choice == 1:
            return self.simulate_checkerboarding(image, n)
        elif choice == 2:
            return image.filter(ImageFilter.GaussianBlur(radius=n))
        elif choice == 3:
            image = self.simulate_checkerboarding(image, n)
            return image.filter(ImageFilter.GaussianBlur(radius=n))
        elif choice == 4:
            image = image.filter(ImageFilter.GaussianBlur(radius=n))
            return self.simulate_checkerboarding(image, n)
        else:
            return image

    # Function to simulate checkerboarding effect
    def simulate_checkerboarding(self, image, scale_factor):
        # Reduce the size of the image by the scale factor
        small_image = image.resize((image.size[0]//scale_factor, image.size[1]//scale_factor), Image.NEAREST)
        # Scale it back up to the original size
        result_image = small_image.resize(image.size, Image.NEAREST)
        return result_image

    def mapped(self, img1, img2):
        # 将 img1 和 img2 分解为三个通道
        img1_r, img1_g, img1_b = img1.split()
        img2_r, img2_g, img2_b = img2.split()

        # 分别对这三个通道进行处理
        img2_r = self.mapped_single_channel(img1_r, img2_r)
        img2_g = self.mapped_single_channel(img1_g, img2_g)
        img2_b = self.mapped_single_channel(img1_b, img2_b)

        # 将处理后的三个通道合并为一张彩色图像
        img2 = Image.merge("RGB", (img2_r, img2_g, img2_b))

        return img2

    def mapped_single_channel(self, img1, img2):
        img1_pixels = np.sort(np.array(img1).flatten())
        img2_pixels = np.sort(np.array(img2).flatten())

        img1_low = float(img1_pixels[int(len(img1_pixels) * 0.05)])
        img1_high = float(img1_pixels[int(len(img1_pixels) * 0.95)])
        img2_low = float(img2_pixels[int(len(img2_pixels) * 0.05)])
        img2_high = float(img2_pixels[int(len(img2_pixels) * 0.95)])

        img2_array = np.array(img2, dtype=float)

        # scale_factor = ((img2_array - img2_low) / (img2_high - img2_low)) * (img1_high - img1_low) + img1_low
        # scale_factor = np.clip(scale_factor, 0, 255, out=scale_factor)

        # # Adding a small epsilon value to the denominator
        epsilon = 1e-5
        scale_factor = ((img2_array - img2_low) / (img2_high - img2_low + epsilon)) * (img1_high - img1_low) + img1_low
        scale_factor = np.clip(scale_factor, 0, 255, out=scale_factor)

        return Image.fromarray(scale_factor.astype(np.uint8))
