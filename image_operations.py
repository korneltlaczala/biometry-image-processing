import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import io

def negative(img):
    img_arr = np.array(img)
    negative_img = 255 - img_arr
    return Image.fromarray(negative_img.astype(np.uint8))

def adjust_brightness(img, factor):
    img_arr = np.array(img)
    img_arr = img_arr * factor
    img_arr = np.clip(img_arr, 0, 255)
    bright_img = Image.fromarray(img_arr.astype(np.uint8))
    return bright_img

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

def binarize(img, threshold=128):
    img_arr = np.array(img)
    binarized_img = np.where(img_arr < threshold, 0, 255)
    return Image.fromarray(binarized_img.astype(np.uint8))

def compute_histogram(img):
    img_arr = np.array(img).flatten()
    hist, bins = np.histogram(img_arr, bins=256, range=[0, 256])

    plt.figure()
    plt.plot(hist, color='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency') 
    plt.title('Image Histogram')
    plt.grid()
    return plt


# filtry

def mean_filter(img, size):
    img_arr = np.array(img)
    h, w = img_arr.shape[0], img_arr.shape[1]
    pad = size // 2
    output = np.zeros_like(img_arr)

    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            window = img_arr[i - pad:i + pad + 1, j - pad:j + pad + 1]
            output[i, j] = np.mean(window, axis=(0, 1))
    return Image.fromarray(output.astype(np.uint8))

def gaussian_filter(img, size, sigma):
    
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma **2)) * np.exp(-((x - (size - 1)/2)**2 + (y - (size - 1)/2)**2) / (2 * sigma ** 2)),
        (size, size)
    )
    kernel = kernel / np.sum(kernel)

    img_arr = np.array(img)
    h, w = img_arr.shape[0], img_arr.shape[1]
    pad = size // 2
    output = np.zeros_like(img_arr)

    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            window = img_arr[i - pad:i + pad + 1, j - pad:j + pad + 1]
            output[i, j] = np.sum(window * kernel[:, :, None], axis=(0, 1))
    
    return Image.fromarray(output.astype(np.uint8))

def sharpen_filter(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    if len(img.shape) == 3:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        r = apply_filter(np.array(r), kernel)
        g = apply_filter(np.array(g), kernel)
        b = apply_filter(np.array(b), kernel)

        sharpened_img = Image.merge('RGB', (Image.fromarray(r), Image.fromarray(g), Image.fromarray(b)))
    else:
        sharpened_img = apply_filter(np.array(img), kernel)

    return sharpened_img

def apply_filter(img, kernel):
    h, w = img.shape[0], img.shape[1]
    pad = kernel.shape[0] // 2

    result = np.zeros_like(img)

    img_padded = np.pad(img, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    for i in range(h):
        for j in range(w):
            region = img_padded[i:i + kernel.shape[0], j:j + kernel.shape[0]]
            result[i, j] = np.sum(region * kernel)
    
    result = np.clip(result, 0, 255)
    return result


