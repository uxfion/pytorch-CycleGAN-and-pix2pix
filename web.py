import streamlit as st
from PIL import Image, ImageFilter
import io

# Dummy function for image quality improvement (replace with your actual model inference)
def improve_image_quality(image, model, parameter):
    # Your image processing code here
    # For now, it just returns the original image
    improved_image = image.filter(ImageFilter.GaussianBlur(radius=parameter))

    return improved_image

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
    model = st.selectbox('Model select', ['Model 1', 'Model 2', 'Model 3'])

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
        improved_image = improve_image_quality(image, model, parameter)

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

