from PIL import Image, ImageFilter
import os
import random
from tqdm import tqdm


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
        new_folder_path = os.path.join(os.path.dirname(self.path), new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        for file_name in tqdm(os.listdir(self.path)):
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
                new_file_path = os.path.join(new_folder_path, new_file_name)
                blurred_image.save(new_file_path)
    
    def blur_pair(self, n_range, method):
        folder_name = os.path.basename(self.path)
        new_folder_name = f"{folder_name}_{method}"
        new_folder_path = os.path.join(os.path.dirname(self.path), new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        for file_name in tqdm(os.listdir(self.path)):
            file_path = os.path.join(self.path, file_name)
            if os.path.isfile(file_path):
                if isinstance(n_range, int):
                    n_B = n_range
                elif isinstance(n_range, tuple) and len(n_range) == 2:
                    n_A = random.randint(min(n_range[0], n_range[1]), max(n_range[0], n_range[1]))
                    n_B = random.randint(min(n_range[0], n_range[1]), n_A)
                else:
                    raise ValueError("Invalid n_range")
                image = Image.open(file_path)
                if method == "Gaussian":
                    blurred_imageA = image.filter(ImageFilter.GaussianBlur(radius=n_A))
                    blurred_imageB = image.filter(ImageFilter.GaussianBlur(radius=n_B))
                elif method == "Box":
                    blurred_imageA = image.filter(ImageFilter.BoxBlur(radius=n_A))
                    blurred_imageB = image.filter(ImageFilter.BoxBlur(radius=n_B))
                elif method == "Median":
                    blurred_imageA = image.filter(ImageFilter.MedianFilter(size=n_A))
                    blurred_imageB = image.filter(ImageFilter.MedianFilter(size=n_B))
                else:
                    raise ValueError("Invalid blur method")
                new_file_name = f"{os.path.splitext(file_name)[0]}_{method}_{n_A}_{n_B}{os.path.splitext(file_name)[1]}"
                new_file_path = os.path.join(new_folder_path, new_file_name)

                aligned_image = Image.new("RGB", (blurred_imageB.size[0] * 2, blurred_imageB.size[1]))
                aligned_image.paste(blurred_imageA, (0, 0))
                aligned_image.paste(blurred_imageB, (blurred_imageA.size[0], 0))
                aligned_image.save(new_file_path)

                


    def blur(self, n_range, method):
        if os.path.isfile(self.path):
            self.blur_single(n_range, method)
        elif os.path.isdir(self.path):
            self.blur_folder(n_range, method)
        else:
            raise ValueError("Invalid path")
        

if __name__ == "__main__":
    image_processor = ImageProcessor("./datasets/coco/raw")
    image_processor.blur_pair((0,20), "Gaussian")
    # for i in tqdm(range(20)):
    #     print(i)
    #     image_processor.blur(i, "Gaussian")
