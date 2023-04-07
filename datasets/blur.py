from PIL import Image, ImageFilter
import os
import random

class ImageProcessor:
    def __init__(self, path):
        self.path = path

    def blur_single(self, n_range, method):
        if isinstance(n_range, int):
            n = n_range
        elif isinstance(n_range, tuple) and len(n_range) == 2:
            n = random.randint(n_range[0], n_range[1])
        else:
            raise ValueError("Invalid n_range")
        image = Image.open(self.path)
        if method == "Gaussian":
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=n))
        elif method == "Box":
            blurred_image = image.filter(ImageFilter.BoxBlur(radius=n))
        elif method == "Median":
            blurred_image = image.filter(ImageFilter.MedianFilter(size=n))
        else:
            raise ValueError("Invalid blur method")
        file_name, ext = os.path.splitext(self.path)
        blurred_image.save(f"{file_name}_{method}_{n}{ext}")

    def blur_folder(self, n_range, method):
        folder_name = os.path.basename(self.path)
        new_folder_name = f"{folder_name}_{method}"
        os.makedirs(new_folder_name, exist_ok=True)
        for file_name in os.listdir(self.path):
            file_path = os.path.join(self.path, file_name)
            if os.path.isfile(file_path):
                if isinstance(n_range, int):
                    n = n_range
                elif isinstance(n_range, tuple) and len(n_range) == 2:
                    n = random.randint(n_range[0], n_range[1])
                else:
                    raise ValueError("Invalid n_range")
                image = Image.open(file_path)
                if method == "Gaussian":
                    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=n))
                elif method == "Box":
                    blurred_image = image.filter(ImageFilter.BoxBlur(radius=n))
                elif method == "Median":
                    blurred_image = image.filter(ImageFilter.MedianFilter(size=n))
                else:
                    raise ValueError("Invalid blur method")
                new_file_name = f"{os.path.splitext(file_name)[0]}_{method}_{n}{os.path.splitext(file_name)[1]}"
                new_file_path = os.path.join(new_folder_name, new_file_name)
                blurred_image.save(new_file_path)

    def blur(self, n_range, method):
        if os.path.isfile(self.path):
            self.blur_single(n_range, method)
        elif os.path.isdir(self.path):
            self.blur_folder(n_range, method)
        else:
            raise ValueError("Invalid path")
        


image_processor = ImageProcessor("./coco/trainB/")
image_processor.blur((0,10), "Gaussian")