import numpy as np


class ModelConfig(object):
    def __init__(self, name, input_name, output_name, input_shape, output_shape,
                 input_data_type=np.float32, output_data_type=np.float32):
        self._name = name
        self._input_name = input_name
        self._output_name = output_name
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._input_data_type = input_data_type
        self._output_data_type = output_data_type

    def name(self):
        return self._name

    def input_name(self):
        return self._input_name

    def output_name(self):
        return self._output_name

    def input_shape(self):
        return self._input_shape

    def output_shape(self):
        return self._output_shape

    def input_data_type(self):
        return self._input_data_type

    def output_data_type(self):
        return self._output_data_type


def get_model_config(model):
    if model == "squeezenet":
        return ModelConfig("squeezenet", "data_0", "softmaxout_1", (1, 3, 224, 224), (1, 1000, 1, 1))
    elif model == "shufflenet":
        return ModelConfig("shufflenet", "gpu_0/data_0", "gpu_0/softmax_1", (1, 3, 224, 224), (1, 1000))
    else:
        return None
