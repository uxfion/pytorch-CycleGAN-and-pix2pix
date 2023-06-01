import os
from PIL import Image

def convert_bmp_to_jpg(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    counter = 1

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".bmp"):
                # 构建输入和输出文件路径
                input_path = os.path.join(root, filename)
                output_filename = f"image_{counter}.jpg"
                output_path = os.path.join(output_folder, output_filename)

                with Image.open(input_path) as img:
                    img = img.convert("RGB")
                    img.save(output_path, "JPEG")

                counter += 1

    print("转换完成！")

input_folder = "./ultrasound_data/ultrasound_crop_delMark"
output_folder = "./ultrasound"

convert_bmp_to_jpg(input_folder, output_folder)

