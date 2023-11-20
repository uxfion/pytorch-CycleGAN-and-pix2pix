import streamlit as st
from PIL import Image, ImageFilter
import io

import torch
from PIL import Image
from torchvision import transforms
from models import create_model
from options.test_options import TestOptions
import numpy as np

model = None
def load_model():
    global model

    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # opt.dataroot="./datasets/xijing/low_quality"
    # opt.name="ultrasound_2023_10_10_batch5"
    # opt.model="my_test"
    # opt.no_dropout=True
    # opt.preprocess="none"
    # opt.gpu_ids=[1]

    # Load the model
    model = create_model(opt)  # You need to set up the `opt` object appropriately
    model.setup(opt)
    model.eval()
    return model

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

# Function to perform inference
def infer(model, image_raw, sigma):
    # Apply necessary transformations
    transform = transforms.Compose([
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

    # Convert to PIL image and return
    fake_image = (fake_image.cpu().squeeze(0) + 1) / 2  # Denormalize
    fake_image = transforms.ToPILImage()(fake_image)


    contrast_image = mapped(image_raw, fake_image)

    return contrast_image

# Dummy function for image quality improvement (replace with your actual model inference)
def improve_image_quality(image, model, parameter):
    # Your image processing code here
    # For now, it just returns the original image
    improved_image = image.filter(ImageFilter.GaussianBlur(radius=parameter))

    return improved_image

if not model:
    print("Loading model...")
    model = load_model()

# ========== web ==========================
# Set page config to make the layout use the full page width
st.set_page_config(layout="wide")

# Title of the webpage
st.title('Image Quality Improvement of Mobile Ultrasound Devices')

col1, col2 = st.columns(2)

# File uploader allows the user to add their own image
with col1:
    uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])

# Dropdown for model selection
with col2:
    model_select = st.selectbox('Model select', ['CycleGAN', 'Model 2', 'Model 3'])

parameter_col, button_col = st.columns(2)
with parameter_col:
    parameter = st.slider('Select parameter', 2, 14, step=2, value=8)
with button_col:
    infer_button = st.button('Infer')

# Button to perform inference
if infer_button:
    if uploaded_file is not None:
        # Open the uploaded image file
        image = Image.open(uploaded_file)

        # Use Streamlit's columns feature to display images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Image before infer', use_column_width=True)

        # Improve image quality
        # improved_image = improve_image_quality(image, model, parameter)
        improved_image = infer(model, image, parameter)

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

