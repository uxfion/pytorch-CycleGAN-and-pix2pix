import streamlit as st
import random
import os
import io
import datetime
from PIL import Image, ImageFilter
import SimpleITK as sitk

import requests

# from cyclegan_infer import load_cyclegan_model, cyclegan_infer

# Set page config to make the layout use the full page width
st.set_page_config(layout="wide")


# @st.cache_resource
# def st_load_cyclegan_model():
#     return load_cyclegan_model()


# cyclegan_model = st_load_cyclegan_model()


# def crop_to_divisible_by_four(img):
#     # Check if dimensions are already divisible by 4
#     if img.width % 4 == 0 and img.height % 4 == 0:
#         return img

#     print(f"图像尺寸{img.size}不能被4整除, 裁剪图像至能被4整除...")
#     # Calculate the new dimensions
#     new_width = img.width - (img.width % 4)
#     new_height = img.height - (img.height % 4)
#     print(f"裁剪后图像尺寸: {new_width} x {new_height}")
#     # Crop the image
#     cropped_img = img.crop((0, 0, new_width, new_height))

#     return cropped_img


def test_infer(model, image, parameter):
    improved_image = image.filter(ImageFilter.GaussianBlur(radius=parameter))
    return improved_image


def random_image_from_folder(folder_path):
    """ 随机从指定文件夹中选择一张图片 """
    files = os.listdir(folder_path)
    random_file = random.choice(files)
    image_path = os.path.join(folder_path, random_file)
    return image_path


# ========== web ==========================
# Title of the webpage
st.title('Image Quality Improvement of Mobile Ultrasound Devices')

col1, col2 = st.columns(2)

# File uploader allows the user to add their own image
with col1:
    uploaded_file = st.file_uploader("Upload Image or DICOM", type=["png", "jpg", "jpeg", "dcm"])

# Dropdown for model selection
with col2:
    model_select = st.selectbox('Model select', ['Super-Resolution Model', 'Model 2', 'Model 3', 'Test'])

parameter_col, button_col = st.columns(2)
with parameter_col:
    parameter = st.slider('Select parameter', 1, 14, step=1, value=8)
with button_col:
    _, demo_col, infer_col, _ = st.columns(4)
    with demo_col:
        demo_button = st.button('Demo')
    with infer_col:
        infer_button = st.button('Infer')

if demo_button:
    demo_file = random_image_from_folder('./datasets/xijing/low_quality/')  # 随机选择图片
    st.session_state['demo_image'] = demo_file
    st.write(f"Pick a random image: {demo_file.replace('xijing/', '')}")
    st.image(demo_file, caption='Random image', width=250)


# Button to perform inference
if infer_button:
    uploaded_file = st.session_state.get('demo_image', uploaded_file)
    if uploaded_file is not None:
        if isinstance(uploaded_file, str):
            with open(uploaded_file, "rb") as f:
                contents = f.read()
        else:
            contents = uploaded_file.getvalue()

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        file = {"file": contents}
        data = {"sigma": parameter}

        # Use Streamlit's columns feature to display images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Image before infer', use_column_width=True)

        # Improve image quality
        # improved_image = improve_image_quality(image, model, parameter)
        # print(f"{datetime.datetime.now()} 推理中...")
        # print("图像信息")
        # print(f"  - 尺寸: {image.size}")
        # print(f"  - 格式: {image.format}")
        # print(f"  - exif: {image.getexif()}")

        # print(f"推理模型: {model_select}")

        if model_select == 'Super-Resolution Model':
            response = requests.post("http://127.0.0.1:1234/process/", files=file, data=data)
            if response.status_code == 200:
                improved_image = Image.open(io.BytesIO(response.content))
            else:
                st.error("Error processing image.")
        elif model_select == 'Model 2':
            response = requests.post("http://127.0.0.1:1111/process/", files=file, data=data)
        elif model_select == 'Model 3':
            response = requests.post("http://127.0.0.1:1111/process/", files=file, data=data)
        elif model_select == 'Test':
            improved_image = test_infer(None, image, parameter)
        else:
            improved_image = image
        

        print(f"{datetime.datetime.now()} 推理完成\n\n\n")

        # Display the image after improvement
        with col2:
            st.image(improved_image, caption='Image after infer', use_column_width=True)

        # Download button
        buf = io.BytesIO()
        improved_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label="Download image",
                           data=byte_im,
                           file_name="improved_image.png",
                           mime="image/png")
    else:
        st.error("Please upload an image to infer.")
