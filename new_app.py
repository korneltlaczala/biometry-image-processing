import streamlit as st
from PIL import Image
import numpy as np
from image_processors import *

def show_image(image, title="image"):
    st.image(image, caption=title, use_container_width=True)

st.title("Image Processing App")

if "processor_flow" not in st.session_state:
    st.session_state.processor_flow = ProcessorFlow()
    st.session_state.grayscale_processor = GrayscaleProcessor()
    st.session_state.brightness_processor = BrightnessProcessor()
    st.session_state.processor_flow.add_processor(st.session_state.grayscale_processor)
    st.session_state.processor_flow.add_processor(st.session_state.brightness_processor)

uploaded_file = st.file_uploader("Choose file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.sidebar.title("Image Operations")
    st.sidebar.subheader("Pixel Operations")

    processor_flow = st.session_state.processor_flow
    grayscale_processor = st.session_state.grayscale_processor
    brightness_processor = st.session_state.brightness_processor

    grayscale = st.sidebar.checkbox("Grayscale", value=grayscale_processor.default_is_enabled)
    brightness_value = st.sidebar.slider("Brightness", -255, 255, brightness_processor.default_value)

    grayscale_processor.set_param("_is_enabled", grayscale)
    brightness_processor.set_param("_value", brightness_value)


    img_processed = processor_flow.process(img)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        show_image(img, title="Original Image")

    modified_img = img.copy()
    with col2:
        st.subheader("Modified Image")
        show_image(img_processed, title="Modified Image")