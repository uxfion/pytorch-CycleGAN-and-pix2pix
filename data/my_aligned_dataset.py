import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFilter
import random
import numpy as np


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

        n_A = random.randint(1, 10)
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

        A = self.random_effect(AB, n_A)
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
        if choice == 1:
            print(f"simulate checkboard {n}")
            return self.simulate_checkerboarding(image, n)
        elif choice == 2:
            print(f"simulate gaussian {n}")
            return image.filter(ImageFilter.GaussianBlur(radius=n))
        elif choice == 3:
            print(f"simulate checkboard gaussian {n}")
            image = self.simulate_checkerboarding(image, n)
            return image.filter(ImageFilter.GaussianBlur(radius=n))
        elif choice == 4:
            print(f"simulate gaussian checkboard {n}")
            image = image.filter(ImageFilter.GaussianBlur(radius=n))
            return self.simulate_checkerboarding(image, n)
        else:
            print(f"no simulate")
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
