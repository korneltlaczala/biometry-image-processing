from PIL import Image
import numpy as np
from image_operations import adjust_brightness

def inspect(img):
    img_arr = np.array(img)
    print(img_arr[:2, :2])

img = Image.open("pudzian.jpg")
inspect(img)
new_img = adjust_brightness(img, -20)
inspect(new_img)



