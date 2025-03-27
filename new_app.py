import streamlit as st
from PIL import Image
import numpy as np
from image_processors import *
from image_operations import compute_histogram, horizontal_projection, vertical_projection, plot_projection
from image_operations import compute_roberts, compute_sobel

def show_image(image, title="image"):
    st.image(image, caption=title, use_container_width=True)

def downscale_image(img):
    max_dim = 1000
    # max_dim = 512
    if img.width < max_dim and img.height < max_dim:
        return img
    if img.width > img.height:
        scale = max_dim / img.width
    else:
        scale = max_dim / img.height
    return img.resize((int(img.width * scale), int(img.height * scale)))

def reset():
    st.session_state.processor_flow.reset()
    st.session_state["exposure_factor"] = st.session_state.exposure_processor.default_factor
    st.session_state["brightness_value"] = st.session_state.brightness_processor.default_value
    st.session_state["contrast_factor"] = st.session_state.contrast_processor.default_factor
    st.session_state["gamma_factor"] = st.session_state.gamma_processor.default_factor
    st.session_state["grayscale"] = st.session_state.grayscale_processor.default_is_enabled
    st.session_state["negative"] = st.session_state.negative_processor.default_is_enabled
    st.session_state["binarization"] = st.session_state.binarization_processor.default_is_enabled
    st.session_state.filter_choice = "None"

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
    st.session_state.gaussian_filter_processor = GaussianFilterProcessor()
    st.session_state.sharpening_filter_processor = SharpeningFilterProcessor()

    st.session_state.processor_flow.add_processor(st.session_state.brightness_processor)
    st.session_state.processor_flow.add_processor(st.session_state.exposure_processor)
    st.session_state.processor_flow.add_processor(st.session_state.contrast_processor)
    st.session_state.processor_flow.add_processor(st.session_state.gamma_processor)
    st.session_state.processor_flow.add_processor(st.session_state.grayscale_processor)
    st.session_state.processor_flow.add_processor(st.session_state.negative_processor)
    st.session_state.processor_flow.add_processor(st.session_state.binarization_processor)
    st.session_state.processor_flow.add_processor(st.session_state.mean_filter_processor)
    st.session_state.processor_flow.add_processor(st.session_state.gaussian_filter_processor)
    st.session_state.processor_flow.add_processor(st.session_state.sharpening_filter_processor)
    
uploaded_file = st.file_uploader("Choose file", type=["jpg", "png", "jpeg"])
downscale = st.checkbox("Downscale", value=True)
try:
    if st.session_state.uploaded_file != uploaded_file:
        st.session_state.processor_flow.reset_cache()
    elif st.session_state.downscale != downscale:
        st.session_state.processor_flow.reset_cache()
except:
    pass
st.session_state.uploaded_file = uploaded_file
st.session_state.downscale = downscale


if uploaded_file is not None:


    img = Image.open(uploaded_file)
    if downscale:
        img = downscale_image(img)

    im_ar = np.array(img, dtype=np.int16)
    st.write(f"Image size: {im_ar.shape}")
    st.sidebar.title("Image Operations")
    st.sidebar.button("Reset", on_click=reset)
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
    gaussian_filter_processor = st.session_state.gaussian_filter_processor
    sharpening_filter_processor = st.session_state.sharpening_filter_processor

    def apply_filter(chosen_filter):

        if "chosen_filter" not in st.session_state:
            st.session_state.chosen_filter = "None"

        if st.session_state.chosen_filter != chosen_filter:
            gaussian_filter_processor.set_param("_is_enabled", False)
            mean_filter_processor.set_param("_is_enabled", False)
            sharpening_filter_processor.set_param("_is_enabled", False)
            if chosen_filter == "Gaussian":
                gaussian_filter_processor.set_param("_is_enabled", True)
                st.sidebar.slider("Size", 3, 15, value=gaussian_filter_processor.size, step=2, key=chosen_filter+"_size")
                st.sidebar.slider("Sigma", 0.1, 5.0, value=gaussian_filter_processor.sigma, key=chosen_filter+"_sigma")
            elif chosen_filter == "Mean":            
                mean_filter_processor.set_param("_is_enabled", True)
                st.sidebar.slider("Size", 3, 15, value=mean_filter_processor.size, step=2, key=chosen_filter+"_size")
            elif chosen_filter == "Sharpening":
                sharpening_filter_processor.set_param("_is_enabled", True)
                if sharpening_filter_processor.type == "basic":
                    index = 0
                else:
                    index = 1
                st.sidebar.slider("Size", 3, 15, value=sharpening_filter_processor.size, step=2, key=chosen_filter+"_size")
                st.sidebar.slider("Strength", 0.01, 2.0, value=sharpening_filter_processor.strength, key=chosen_filter+"_strength")
                st.sidebar.radio("Type", ["basic", "strong"], horizontal=True, index = index, key=chosen_filter+"_type")

            st.session_state.chosen_filter = chosen_filter
            
        elif chosen_filter != "None":
            size = st.session_state[chosen_filter+"_size"]
            if chosen_filter == "Gaussian":
                sigma = st.session_state[chosen_filter+"_sigma"]
                gaussian_filter_processor.set_param("_size", size)
                gaussian_filter_processor.set_param("_sigma", sigma)

            elif chosen_filter == "Mean":
                mean_filter_processor.set_param("_size", size)

            elif chosen_filter == "Sharpening":
                strength = st.session_state[chosen_filter+"_strength"]
                type = st.session_state[chosen_filter+"_type"].lower()
                sharpening_filter_processor.set_param("_size", size)
                sharpening_filter_processor.set_param("_strength", strength)
                sharpening_filter_processor.set_param("_type", type)


            st.sidebar.slider("Size", 3, 15, value=size, step=2, key=chosen_filter+"_size")
            if chosen_filter == "Gaussian":
                st.sidebar.slider("Sigma", 0.1, 5.0, value=sigma, key=chosen_filter+"_sigma")
            if chosen_filter == "Sharpening":
                if type == "basic":
                    index = 0
                elif type == "strong":
                    index = 1
                st.sidebar.slider("Strength", 0.01, 2.0, value=strength, key=chosen_filter+"_strength")
                st.sidebar.radio("Type", ["basic", "strong"], horizontal=True, index = index, key=chosen_filter+"_type")

    try:
        exposure_processor.set_param("_factor", st.session_state.exposure_factor)
        brightness_processor.set_param("_value", st.session_state.brightness_value)
        contrast_processor.set_param("_factor", st.session_state.contrast_factor)
        gamma_processor.set_param("_factor", st.session_state.gamma_factor)
        grayscale_processor.set_param("_is_enabled", st.session_state.grayscale)
        negative_processor.set_param("_is_enabled", st.session_state.negative)
        binarization_processor.set_param("_is_enabled", st.session_state.binarization)
        if st.session_state.binarization:
            binarization_processor.set_param("_threshold", st.session_state.binarization_threshold)
    except:
        pass

    st.sidebar.slider("Exposure", 0.1, 3.0, value=exposure_processor.factor, key="exposure_factor")
    st.sidebar.slider("Brightness", -255, 255, value=brightness_processor.value, key="brightness_value")
    st.sidebar.slider("Contrast", 0.1, 3.0, value=contrast_processor.factor, key="contrast_factor")
    st.sidebar.slider("Gamma", 0.1, 3.0, value=gamma_processor.factor, key="gamma_factor")
    st.sidebar.checkbox("Grayscale", value=grayscale_processor.is_enabled, key="grayscale")
    st.sidebar.checkbox("Negative", value=negative_processor.is_enabled, key="negative")
    st.sidebar.checkbox("Binarization", value=binarization_processor.is_enabled, key="binarization")
    if binarization_processor.is_enabled:
        st.sidebar.slider("Binarization Threshold", 0, 255, value=binarization_processor.threshold, key="binarization_threshold")
    options = ["None", "Gaussian", "Mean", "Sharpening"]
    st.sidebar.radio("Filter", options, index=0, key="filter_choice")
    apply_filter(st.session_state.filter_choice)

    img_processed = processor_flow.process(img)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        show_image(img, title="Original Image")

    modified_img = img.copy()
    with col2:
        st.subheader("Modified Image")
        show_image(img_processed, title="Modified Image")

    st.sidebar.subheader("Edge detection")
    show_roberts = st.sidebar.checkbox("Roberts cross")
    show_sobel = st.sidebar.checkbox("Sobel operator")
    edge_detection_threshold = st.sidebar.slider("Threshold", 0, 255, 100, key="edge_detection_threshold")

    if show_roberts:
        col1, col2 = st.columns(2)
        with col1:
            roberts_original = compute_roberts(img, edge_detection_threshold)
            show_image(roberts_original, title="Roberts cross of Original Image")
        with col2:
            roberts_processed = compute_roberts(img_processed, edge_detection_threshold)
            show_image(roberts_processed, title="Roberts cross of Modified Image")

    if show_sobel:
        col1, col2 = st.columns(2)
        with col1:
            sobel_original = compute_sobel(img, edge_detection_threshold)
            show_image(sobel_original, title="Sobel operator of Original Image")
        with col2:
            sobel_proocessed = compute_sobel(img_processed, edge_detection_threshold)
            show_image(sobel_proocessed, title="Sobel operator of Modified Image")

        
    st.sidebar.subheader("Histogram")
    show_histogram = st.sidebar.checkbox("Show histograms")

    if show_histogram:
        col1, col2 = st.columns(2)
        
        with col1:
            plt = compute_histogram(img, sigma=True)
            st.pyplot(plt)
        with col2:
            plt = compute_histogram(img_processed, sigma=True)
            st.pyplot(plt)
    
    st.sidebar.subheader("Projection")
    show_projection = st.sidebar.checkbox("Show projections")

    if show_projection:
        st.subheader("Projections of Modified Image")
        col1, col2 = st.columns(2)
        with col1:
            projection = horizontal_projection(img_processed)
            plt = plot_projection(projection, 'Horizontal')
            st.pyplot(plt)
        with col2:
            projection = vertical_projection(img_processed)
            plt = plot_projection(projection, 'Vertical')
            st.pyplot(plt)
