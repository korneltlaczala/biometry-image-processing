import streamlit as st
from PIL import Image
import numpy as np
from image_processors import *

def show_image(image, title="image"):
    st.image(image, caption=title, use_container_width=True)

st.title("Image Processing App")

if "processor_flow" not in st.session_state:
    st.session_state.processor_flow = ProcessorFlow()
    st.session_state.brightness_processor = BrightnessProcessor()
    st.session_state.exposure_processor = ExposureProcessor()
    st.session_state.contrast_processor = ContrastProcessor()
    st.session_state.gamma_processor = GammaProcessor()
    st.session_state.grayscale_processor = GrayscaleProcessor()
    st.session_state.negative_processor = NegativeProcessor()
    st.session_state.binarization_processor = BinarizationProcessor()

    st.session_state.processor_flow.add_processor(st.session_state.brightness_processor)
    st.session_state.processor_flow.add_processor(st.session_state.exposure_processor)
    st.session_state.processor_flow.add_processor(st.session_state.contrast_processor)
    st.session_state.processor_flow.add_processor(st.session_state.gamma_processor)
    st.session_state.processor_flow.add_processor(st.session_state.grayscale_processor)
    st.session_state.processor_flow.add_processor(st.session_state.negative_processor)
    st.session_state.processor_flow.add_processor(st.session_state.binarization_processor)
    
uploaded_file = st.file_uploader("Choose file", type=["jpg", "png", "jpeg"])
try:
    if st.session_state.uploaded_file != uploaded_file:
        st.session_state.processor_flow.reset_cache()
except:
    pass
st.session_state.uploaded_file = uploaded_file

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.sidebar.title("Image Operations")
    st.sidebar.subheader("Pixel Operations")

    processor_flow = st.session_state.processor_flow
    exposure_processor = st.session_state.exposure_processor
    brightness_processor = st.session_state.brightness_processor
    contrast_processor = st.session_state.contrast_processor
    gamma_processor = st.session_state.gamma_processor
    grayscale_processor = st.session_state.grayscale_processor
    negative_processor = st.session_state.negative_processor
    binarization_processor = st.session_state.binarization_processor

    exposure_factor = st.sidebar.slider("Exposure", 0.1, 3.0, exposure_processor.default_factor)
    brightness_value = st.sidebar.slider("Brightness", -255, 255, brightness_processor.default_value)
    contrast_factor = st.sidebar.slider("Contrast", 0.1, 3.0, contrast_processor.default_factor)
    gamma_factor = st.sidebar.slider("Gamma", 0.1, 3.0, gamma_processor.default_factor)
    grayscale = st.sidebar.checkbox("Grayscale", value=grayscale_processor.default_is_enabled)
    negative = st.sidebar.checkbox("Negative", value=negative_processor.default_is_enabled)
    binarization = st.sidebar.checkbox("Binarization", value=binarization_processor.default_is_enabled)
    binarization_threshold = st.sidebar.slider("Binarization Threshold", 0, 255, binarization_processor.default_threshold)

    exposure_processor.set_param("_factor", exposure_factor)
    brightness_processor.set_param("_value", brightness_value)
    contrast_processor.set_param("_factor", contrast_factor)
    gamma_processor.set_param("_factor", gamma_factor)
    grayscale_processor.set_param("_is_enabled", grayscale)
    negative_processor.set_param("_is_enabled", negative)
    binarization_processor.set_param("_is_enabled", binarization)
    binarization_processor.set_param("_threshold", binarization_threshold)

    img_processed = processor_flow.process(img)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        show_image(img, title="Original Image")

    modified_img = img.copy()
    with col2:
        st.subheader("Modified Image")
        show_image(img_processed, title="Modified Image")
    