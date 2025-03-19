import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import io

def negative(img):
    img_arr = np.array(img)
    negative_img = 255 - img_arr
    return Image.fromarray(negative_img.astype(np.uint8))

def adjust_exposure(img, factor):
    img_arr = np.array(img)
    img_arr = img_arr * factor
    img_arr = np.clip(img_arr, 0, 255)
    exposure_img = Image.fromarray(img_arr.astype(np.uint8))
    return exposure_img

def adjust_brightness(img, value):
    img_arr = np.array(img, dtype=np.int32)
    brightness_img = img_arr + value
    brightness_img = np.clip(brightness_img, 0, 255)
    return Image.fromarray(brightness_img.astype(np.uint8))

def convert_to_grayscale(img):
    img_arr = np.array(img)  
    if len(img_arr.shape) == 3:  
        gray_img = np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])  
        return Image.fromarray(gray_img.astype(np.uint8))  
    else:
        return img 
    
def adjust_contrast(img, factor):
    img_arr = np.array(img)
    mean = np.mean(img_arr)
    contrast_img = np.clip((img_arr - mean) * factor + mean, 0, 255)
    return Image.fromarray(contrast_img.astype(np.uint8))

def adjust_gamma(img, gamma):
    img_arr = np.array(img)
    gamma_img = np.power(img_arr / 255.0, gamma) * 255
    return Image.fromarray(gamma_img.astype(np.uint8))

def binarize(img, threshold=128):
    img_arr = np.array(img)
    binarized_img = np.where(img_arr < threshold, 0, 255)
    return Image.fromarray(binarized_img.astype(np.uint8))

def compute_histogram(img):
    img_arr = np.array(img)

    if len(img_arr.shape) == 2:
        colors = ['black']
        hist_data = img_arr
    else:
        colors = ['red', 'green', 'blue']
        hist_data = [img_arr[:, :, i] for i in range(3)]
    
    plt.figure()
    for data, color in zip(hist_data, colors):
        hist, bins = np.histogram(data.flatten(), bins=256, range=[0, 256])
        plt.fill_between(bins[:-1], hist, color=color, alpha=0.5)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Image Histogram')
    plt.grid()
    return plt
   

# filtry

def mean_filter(img, size):
    img_arr = np.array(img)

    kernel = np.ones((size, size)) / (size ** 2)

    output = apply_filter(img_arr, kernel)

    return Image.fromarray(output.astype(np.uint8))

def gaussian_filter(img, size, sigma):
    
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma **2)) * np.exp(-((x - (size - 1)/2)**2 + (y - (size - 1)/2)**2) / (2 * sigma ** 2)),
        (size, size)
    )
    print(kernel)
    kernel = kernel / np.sum(kernel)

    img_arr = np.array(img)

    output = apply_filter(img_arr, kernel)
    
    return Image.fromarray(output.astype(np.uint8))

def sharpen_filter(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])

    img_arr = np.array(img)
    img_arr = np.array(img_arr, dtype=np.int16)
    sharpened_img = apply_filter(img_arr, kernel)

    return Image.fromarray(sharpened_img.astype(np.uint8))

def apply_filter(img, kernel):
    h, w = img.shape[0], img.shape[1]
    pad = kernel.shape[0] // 2

    result = np.zeros_like(img)

    if len(img.shape) == 3:
        chanels = img.shape[2]
    else:
        chanels = 1
        img = img[:, :, None]

    img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    # print(f"Original image shape: {img.shape}")
    # print(f"Padded image shape: {img_padded.shape}")

    for i in range(h):
        for j in range(w):
            for c in range(chanels):
                window = img_padded[i:i + kernel.shape[0], j:j + kernel.shape[0], c]
                result[i, j, c] = np.sum(window * kernel)
    
    result = np.clip(result, 0, 255)
    return result

