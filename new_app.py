import streamlit as st
from PIL import Image
import numpy as np
from image_processors import *
from image_operations import compute_histogram, horizontal_projection, vertical_projection, plot_projection

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
    st.session_state.mean_filter_processor = MeanFilterProcessor()

    st.session_state.processor_flow.add_processor(st.session_state.brightness_processor)
    st.session_state.processor_flow.add_processor(st.session_state.exposure_processor)
    st.session_state.processor_flow.add_processor(st.session_state.contrast_processor)
    st.session_state.processor_flow.add_processor(st.session_state.gamma_processor)
    st.session_state.processor_flow.add_processor(st.session_state.grayscale_processor)
    st.session_state.processor_flow.add_processor(st.session_state.negative_processor)
    st.session_state.processor_flow.add_processor(st.session_state.binarization_processor)
    st.session_state.processor_flow.add_processor(st.session_state.mean_filter_processor)
    
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
    mean_filter_processor = st.session_state.mean_filter_processor


    exposure_factor = st.sidebar.slider("Exposure", 0.1, 3.0, exposure_processor.default_factor)
    brightness_value = st.sidebar.slider("Brightness", -255, 255, brightness_processor.default_value)
    contrast_factor = st.sidebar.slider("Contrast", 0.1, 3.0, contrast_processor.default_factor)
    gamma_factor = st.sidebar.slider("Gamma", 0.1, 3.0, gamma_processor.default_factor)
    grayscale = st.sidebar.checkbox("Grayscale", value=grayscale_processor.default_is_enabled)
    negative = st.sidebar.checkbox("Negative", value=negative_processor.default_is_enabled)
    binarization = st.sidebar.checkbox("Binarization", value=binarization_processor.default_is_enabled)
    binarization_threshold = st.sidebar.slider("Binarization Threshold", 0, 255, binarization_processor.default_threshold)
    mean_filter_kernel_size = st.sidebar.slider("Mean Filter Kernel Size", 3, 15, mean_filter_processor.default_size, step=2)

    exposure_processor.set_param("_factor", exposure_factor)
    brightness_processor.set_param("_value", brightness_value)
    contrast_processor.set_param("_factor", contrast_factor)
    gamma_processor.set_param("_factor", gamma_factor)
    grayscale_processor.set_param("_is_enabled", grayscale)
    negative_processor.set_param("_is_enabled", negative)
    binarization_processor.set_param("_is_enabled", binarization)
    binarization_processor.set_param("_threshold", binarization_threshold)
    mean_filter_processor.set_param("_size", mean_filter_kernel_size)

    img_processed = processor_flow.process(img)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        show_image(img, title="Original Image")

    modified_img = img.copy()
    with col2:
        st.subheader("Modified Image")
        show_image(img_processed, title="Modified Image")

        
    st.sidebar.subheader("Histogram")
    show_histogram = st.sidebar.checkbox("Show histograms")

    if show_histogram:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            plt = compute_histogram(img)
            st.pyplot(plt)
        with col2:
            st.subheader("Modified Image")
            plt = compute_histogram(img_processed)
            st.pyplot(plt)
    
    st.sidebar.subheader("Projection")
    show_projection = st.sidebar.checkbox("Show projections")

    if show_projection:
        st.subheader("Projections of Modified Image")
        col1, col2 = st.columns(2)
        with col1:
            #st.subheader("Horizontal Projection")
            projection = horizontal_projection(img_processed)
            plt = plot_projection(projection, 'Horizontal')
            st.pyplot(plt)
        with col2:
            #st.subheader("Vertical Projection")
            projection = vertical_projection(img_processed)
            plt = plot_projection(projection, 'Vertical')
            st.pyplot(plt)
