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


class Processor:
    def __init__(self):
        self.changed_params = True
        self.last_img = None

    def process(self, img):
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


class GrayscaleProcessor(Processor):

    def __init__(self,):
        super().__init__()
        self.default_is_enabled = False
        self._is_enabled = self.default_is_enabled
         
    def _process(self, img_arr):
        self.changed_params = False
        if not self._is_enabled:
            return img_arr
        return np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])  

    def _process_pixelwise(self, img_arr):
        return img_arr

    @property
    def is_enabled(self):
        return self._is_enabled

class BrightnessProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_value = 0
        self._value = self.default_value

    def _process(self, img_arr):
        self.changed_params = False
        img_arr = np.array(img_arr, dtype=np.int16)
        img_arr = np.clip(img_arr + self._value, 0, 255)
        return img_arr

    def _process_pixelwise(self, img_arr):  
        return img_arr

    @property
    def value(self):
        return self._value