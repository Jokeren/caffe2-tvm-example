from __future__ import absolute_import
import numpy as np


class DataType(object):
    def __init__(self, name, np_type, tvm_type):
        self._name = name
        self._np_type = np_type
        self._tvm_type = tvm_type

    def name(self):
        return self._name

    def np_type(self):
        return self._np_type

    def tvm_type(self):
        return self._tvm_type


class CompositeDataType(object):
    float32 = DataType(["float32", "float"], np.float32, "float32")
    int32 = DataType(["int", "int32"], np.int32, "int32")
    int8 = DataType(["int8", "char"], np.int8, "int8")


def get_data_type(name):
    if name in CompositeDataType.float32.name():
        return CompositeDataType.float32
    elif name in CompositeDataType.int32.name():
        return CompositeDataType.int32
    elif name in CompositeDataType.int8.name():
        return CompositeDataType.int8
    else:
        return None
