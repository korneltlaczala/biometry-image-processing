import numpy as np
from PIL import Image

class ProcessorFlow:
    def __init__(self):
        self.processors = []

    def add_processor(self, processor):
        self.processors.append(processor)

    def process(self, img):
        changed_params = False
        for processor in self.processors:
            if processor.changed_params:
                changed_params = True
            if not changed_params:
                img = processor.get_last_img()
                continue
            img = processor.process(img)
        return img

    def reset_cache(self):
        for processor in self.processors:
            processor.changed_params = True


class Processor:
    def __init__(self):
        self.changed_params = True
        self.last_img = None

    def process(self, img):
        self.changed_params = False
        img_arr = np.array(img)
        img_arr = self._process(img_arr)
        self.last_img = Image.fromarray(img_arr.astype(np.uint8))
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
        return self.last_img


class BrightnessProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_value = 0
        self._value = self.default_value

    def _process(self, img_arr):
        img_arr = np.array(img_arr, dtype=np.int16)
        img_arr = np.clip(img_arr + self._value, 0, 255)
        return img_arr

    def _process_pixelwise(self, img_arr):  
        raise NotImplementedError

    @property
    def value(self):
        return self._value


class ExposureProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_factor = 1.0
        self._factor = self.default_factor

    def _process(self, img_arr):
        img_arr = np.array(img_arr, dtype=np.int16)
        img_arr = np.clip(img_arr * self._factor, 0, 255)
        return img_arr

    def _process_pixelwise(self, img_arr):  
        img_arr = np.array(img_arr, dtype=np.int16)

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
        img_arr = np.array(img_arr, dtype=np.int16)
        mean = np.mean(img_arr)
        img_arr = np.clip((img_arr - mean) * self._factor + mean, 0, 255)
        return img_arr

    def _process_pixelwise(self, img_arr):  
        raise NotImplementedError
    
    @property
    def factor(self):
        return self._factor
    

class GammaProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_factor = 1.0
        self._factor = self.default_factor

    def _process(self, img_arr):
        img_arr = np.array(img_arr, dtype=np.int16)
        img_arr = np.power(img_arr / 255.0, self._factor) * 255
        return img_arr

    def _process_pixelwise(self, img_arr):  
        raise NotImplementedError
    
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
        raise NotImplementedError

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
        raise NotImplementedError
    
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
        img_arr = np.array(img_arr, dtype=np.int16)
        img_arr = np.where(img_arr > self._threshold, 255, 0)
        return img_arr

    def _process_pixelwise(self, img_arr):  
        raise NotImplementedError
    
    @property
    def threshold(self):
        return self._threshold


class MeanFilterProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_size = 3
        self._size = self.default_size

    def _process(self, img_arr):
        img_arr = np.array(img_arr, dtype=np.int16)
        kernel = MeanKernel(self.size)

        img_arr = kernel.convolute(img_arr)

        return img_arr
    
    @property
    def size(self):
        return self._size

class GaussianFilterProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_size = 3
        self._size = self.default_size

    def _process(self, img_arr):
        img_arr = np.array(img_arr, dtype=np.int16)
        

        return img_arr
    @property
    def size(self):
        return self._size

class SharpeningFilterProcessor(Processor):

    def __init__(self):
        super().__init__
        self.default_size = 3
        self._size = self.default_size

    def _process(self, img_arr):
        img_arr = np.array(img_arr, dtype=np.int16)

        return img_arr
    
    @property
    def size(self):
        return self._size

class Kernel():

    def __init__(self):
        self.kernel = None

    def convolute(self, img_arr):
        h, w = img_arr.shape[0], img_arr.shape[1]
        pad = self.kernel.shape[0] // 2

        result = np.zeros_like(img_arr)

        if len(img_arr.shape) == 3:
            chanels = img_arr.shape[2]
        else:
            chanels = 1
            img_arr = img_arr[:, :, None]

        img_padded = np.pad(img_arr, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

        for i in range(h):
            for j in range(w):
                for c in range(chanels):
                    window = img_padded[i:i + self.kernel.shape[0], j:j + self.kernel.shape[0], c]
                    result[i, j, c] = np.sum(window * self.kernel)
        
        result = np.clip(result, 0, 255)
        return result

# class SharpeningKernel(Kernel):

#     def __init__(self, size):
        

# class StrongSharpeningKernel(Kernel):

#     def __init__(self, size):
#         pass

class MeanKernel(Kernel):

    def __init__(self, size):
        self.kernel = np.ones((size, size))
        self.kernel = self.kernel / np.sum(self.kernel)

class GaussianBlurKernel(Kernel):

    def __init__(self, size):
        self.kernel = np.fromfunction(
            lambda x, y: (1/ (2 * np.pi * size **2)) * np.exp(-((x - (size - 1)/2)**2 + (y - (size - 1)/2)**2) / (2 * size ** 2)),
            (size, size)
        )
        self.kernel = self.kernel / np.sum(self.kernel)