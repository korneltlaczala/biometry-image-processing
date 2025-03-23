import numpy as np
from PIL import Image

class ProcessorFlow:
    def __init__(self):
        self.processors = []

    def add_processor(self, processor):
        self.processors.append(processor)

    def process(self, img):
        print("-" * 20)
        print("Processing image...")
        changed_params = False
        for processor in self.processors:
            if processor.changed_params:
                changed_params = True
            if not changed_params:
                img = processor.get_last_img()
                continue
            img = processor.process(img)
        print("Processing finished")
        print("-" * 20)
        return img

    def reset_cache(self):
        for processor in self.processors:
            processor.changed_params = True


class Processor:
    def __init__(self):
        self.changed_params = True
        self.last_img = None

    def process(self, img):
        print(f"calculating image for: {self.__class__.__name__}")
        self.changed_params = False
        img_arr = np.array(img, dtype=np.int32)
        img_arr = self._process(img_arr).astype(np.uint8)
        self.last_img = Image.fromarray(img_arr)
        return self.last_img

    def _process(self, img_arr):
        raise NotImplementedError

    def _process_pixelwise(self, img_arr):
        raise NotImplementedError

    def set_param(self, param_name, value):
        if getattr(self, param_name) == value:
            return
        self.changed_params = True
        setattr(self, param_name, value)

    def get_last_img(self):
        print(f"Using cached image for: {self.__class__.__name__}")
        return self.last_img


class BrightnessProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_value = 0
        self._value = self.default_value

    def _process(self, img_arr):
        img_arr = np.clip(img_arr + self._value, 0, 255)
        return img_arr

    def _process_pixelwise(self, img_arr):  
        if len(img_arr.shape) ==3:
            channels = 3
        else:
            channels = 1
            img_arr = img_arr[:, :, None]

        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(channels):
                    img_arr[i, j, c] = np.clip(img_arr[i, j, c] + self._value, 0, 255)
        return img_arr

    @property
    def value(self):
        return self._value


class ExposureProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_factor = 1.0
        self._factor = self.default_factor

    def _process(self, img_arr):
        img_arr = np.clip(img_arr * self._factor, 0, 255)
        return img_arr

    def _process_pixelwise(self, img_arr):  

        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(img_arr.shape[2]):
                    img_arr[i, j, c] = np.clip(img_arr[i, j, c] * self._factor, 0, 255)
        return img_arr
    @property
    def factor(self):
        return self._factor


class ContrastProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_factor = 1.0
        self._factor = self.default_factor

    def _process(self, img_arr):
        mean = np.mean(img_arr)
        img_arr = np.clip((img_arr - mean) * self._factor + mean, 0, 255)
        return img_arr

    def _process_pixelwise(self, img_arr):  

        if len(img_arr.shape) == 3:
            channels = 3
        else:
            channels = 1
            img_arr = img_arr[:, :, None]
        
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(channels):
                    mean = np.mean(img_arr[i, j, c])
                    img_arr[i, j, c] = np.clip((img_arr[i, j, c] - mean) * self._factor + mean, 0, 255)
        return img_arr
    
    @property
    def factor(self):
        return self._factor
    

class GammaProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_factor = 1.0
        self._factor = self.default_factor

    def _process(self, img_arr):
        img_arr = np.power(img_arr / 255.0, self._factor) * 255
        return img_arr

    def _process_pixelwise(self, img_arr):  
        if len(img_arr.shape) == 3:
            channels = 3
        else:
            channels = 1
            img_arr = img_arr[:, :, None]
        
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(channels):
                    img_arr[i, j, c] = np.power(img_arr[i, j, c] / 255.0, self._factor) * 255
        return img_arr
    
    @property
    def factor(self):
        return self._factor


class GrayscaleProcessor(Processor):

    def __init__(self,):
        super().__init__()
        self.default_is_enabled = False
        self._is_enabled = self.default_is_enabled
         
    def _process(self, img_arr):
        if not self._is_enabled:
            return img_arr
        img_arr = np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])  
        return img_arr

    def _process_pixelwise(self, img_arr):
        if len(img_arr.shape) == 3:
            channels = 3
        else:
            return img_arr
        
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(channels):
                    img_arr[i, j, c] = np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])
        return img_arr

    @property
    def is_enabled(self):
        return self._is_enabled


class NegativeProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_is_enabled = False
        self._is_enabled = self.default_is_enabled

    def _process(self, img_arr):
        if not self._is_enabled:
            return img_arr
        img_arr = 255 - img_arr
        return img_arr
    
    def _process_pixelwise(self, img_arr):  
        if len(img_arr.shape) == 3:
            channels = 3
        else:
            channels = 1
            img_arr = img_arr[:, :, None]
        
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(channels):
                    img_arr[i, j, c] = img_arr[i, j, c] - 255
        return img_arr
    
    @property
    def is_enabled(self):
        return self._is_enabled


class BinarizationProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_is_enabled = False
        self._is_enabled = self.default_is_enabled
        self.default_threshold = 128
        self._threshold = self.default_threshold

    def _process(self, img_arr):
        if not self._is_enabled:
            return img_arr
        img_arr = np.where(img_arr > self._threshold, 255, 0)
        return img_arr

    def _process_pixelwise(self, img_arr):  
        if not self._is_enabled:
            return img_arr
        if len(img_arr.shape) == 3:
            channels = 3
        else:
            channels = 1
            img_arr = img_arr[:, :, None]
        
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(channels):
                    img_arr[i, j, c] = np.where(img_arr[i, j, c] > self._threshold, 255, 0)
        return img_arr
    
    @property
    def threshold(self):
        return self._threshold


class FilterProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_is_enabled = False
        self._is_enabled = self.default_is_enabled
        self.default_size = 3
        self._size = self.default_size

    @property
    def size(self):
        return self._size


class MeanFilterProcessor(FilterProcessor):

    def _process(self, img_arr):
        if not self._is_enabled:
            return img_arr

        kernel = MeanKernel(self.size)
        img_arr = kernel.convolute(img_arr)
        return img_arr
    

class GaussianFilterProcessor(FilterProcessor):

    def __init__(self):
        super().__init__()
        self.default_sigma = 1.0
        self._sigma = self.default_sigma

    def _process(self, img_arr):
        if not self._is_enabled:
            return img_arr

        kernel = GaussianBlurKernel(self.size, self.sigma)
        img_arr = kernel.convolute(img_arr)
        return img_arr

    @property
    def sigma(self):
        return self._sigma


class SharpeningFilterProcessor(FilterProcessor):

    def __init__(self):
        super().__init__()
        self.default_strength = 0.1
        self._strength = self.default_strength
        self.default_type = "basic"
        self._type = self.default_type

    def _process(self, img_arr):
        if not self._is_enabled:
            return img_arr

        if self.type == "basic":
            kernel = SharpeningKernel(self.size, self.strength)
        elif self.type == "strong":
            kernel = StrongSharpeningKernel(self.size, self.strength)
        img_arr = kernel.convolute(img_arr)
        return img_arr
    
    @property
    def strength(self):
        return self._strength
    
    @property
    def type(self):
        return self._type

class Kernel():

    def __init__(self):
        self.kernel = None

    def convolute(self, img_arr):
        h, w = img_arr.shape[0], img_arr.shape[1]
        pad = self.kernel.shape[0] // 2

        result = np.zeros_like(img_arr)

        if len(img_arr.shape) == 3:
            chanels = img_arr.shape[2]
            grayscale = False
            img_padded = np.pad(img_arr, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        else:
            grayscale = True
            img_padded = np.pad(img_arr, ((pad, pad), (pad, pad)), mode='reflect')


        for i in range(h):
            for j in range(w):
                if grayscale:
                    window = img_padded[i:i + self.kernel.shape[0], j:j + self.kernel.shape[0]]
                    result[i, j] = np.sum(window * self.kernel)
                    continue
                for c in range(chanels):
                    window = img_padded[i:i + self.kernel.shape[0], j:j + self.kernel.shape[0], c]
                    result[i, j, c] = np.sum(window * self.kernel)
        
        result = np.clip(result, 0, 255)
        return result


class MeanKernel(Kernel):
    def __init__(self, size):
        self.kernel = np.ones((size, size))
        self.kernel = self.kernel / np.sum(self.kernel)


class GaussianBlurKernel(Kernel):
    def __init__(self, size, sigma):
        self.kernel = np.fromfunction(
            lambda x, y: (1/ (2 * np.pi * sigma **2)) * np.exp(-((x - (size - 1)/2)**2 + (y - (size - 1)/2)**2) / (2 * sigma ** 2)),
            (size, size)
        )
        self.kernel = self.kernel / np.sum(self.kernel)

class SharpeningKernel(Kernel):
    def __init__(self, size, strength):
        mid = size // 2
        self.kernel = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dist = abs(i - mid) + abs(j - mid)
                self.kernel[i, j] = min(0, dist - mid - 1)
        
        desired_sum = 4*strength
        self.kernel[mid, mid] = 0
        self.kernel[mid, mid] = -np.sum(self.kernel)
        scale = desired_sum / self.kernel[mid, mid]
        print(f"desired sum: {desired_sum}")
        print(f"scale: {scale}")
        self.kernel = self.kernel * scale
        self.kernel[mid, mid] += 1
        print(self.kernel)

class StrongSharpeningKernel(Kernel):
    def __init__(self, size, strength):
        mid = size // 2
        self.kernel = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dist = max(abs(i - mid), abs(j - mid))
                self.kernel[i, j] = min(0, dist - mid - 1)
        
        desired_sum = 4*strength
        self.kernel[mid, mid] = 0
        self.kernel[mid, mid] = -np.sum(self.kernel)
        scale = desired_sum / self.kernel[mid, mid]
        self.kernel = self.kernel * scale
        self.kernel[mid, mid] += 1
        print(self.kernel)
