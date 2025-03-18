import streamlit as st
from PIL import Image
import numpy as np
from image_operations import convert_to_grayscale
from image_operations import negative
from image_operations import adjust_brightness, adjust_exposure, adjust_contrast, adjust_gamma, binarize, compute_histogram
from image_operations import mean_filter, mean_filter, gaussian_filter, sharpen_filter
# from image_operations import convert_image_to_bytes

def show_image(image, title="image"):
    st.image(image, caption=title, use_container_width=True)

st.title("Image Processing App")

uploaded_file = st.file_uploader("Choose file", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = np.array(img)

    st.sidebar.title("Image Operations")
    st.sidebar.subheader("Pixel Operations")
    operation = st.sidebar.selectbox("Choose option", [
        "Grayscale",
        "Exposure",
        "Brightness",
        "Contrast",
        "Gamma",
        "Negative",
        "Binarization",
        "None"
    ])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        show_image(img, title="Original Image")

    modified_img = img.copy()

    if operation == "Grayscale":
        result = convert_to_grayscale(modified_img)
    elif operation == "Exposure":
        factor = st.sidebar.slider("Exposure", 0.1, 3.0, 1.0)
        result = adjust_exposure(img, factor)
    elif operation == "Brightness":
        value = st.sidebar.slider("Brightness", -255, 255, 0)
        result = adjust_brightness(img, value)
    elif operation == "Contrast":
        factor = st.sidebar.slider("Kontrast", 0.0, 3.0, 1.0)
        result = adjust_contrast(img, factor)
    elif operation == "Gamma":
        gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0)
        result = adjust_gamma(img, gamma)
    elif operation == "Negative":
        result = negative(img)
    elif operation == "Binarization":
        threshold = st.sidebar.slider("Próg binarizacji", 0, 255, 128)
        result = binarize(img, threshold)
    elif operation == "None":
        result = img

    st.sidebar.subheader('Filters')
    apply_filter = st.sidebar.toggle("Apply filter", False)
    if apply_filter:
        filter = st.sidebar.selectbox("Choose filter", ['gaussian', 'mean', 'sharpening'])
        if filter == 'mean':
            size = st.sidebar.slider("Size", 3, 15, 3, step=2)
            result = mean_filter(img, size)
        elif filter == 'gaussian':
            size = st.sidebar.slider("Size", 3, 21, 3, step=2)
            sigma = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0)
            result = gaussian_filter(img, size, sigma)
        elif filter == 'sharpening':
            result = sharpen_filter(img)

    with col2:
        st.subheader("Modified Image")
        show_image(result, title="Modified Image")

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
            plt = compute_histogram(result)
            st.pyplot(plt)
