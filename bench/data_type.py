from __future__ import absolute_import
import numpy as np


class DataType(object):
    def __init__(self, name, np_type, tvm_type):
        self._name = name
        self._np_type = np_type
        self._tvm_type = tvm_type
        return

    def name(self):
        return self._name

    def np_type(self):
        return self._np_type

    def tvm_type(self):
        return self._tvm_type


def get_data_type(name):
    if name == "int" or name == "int32":
        return DataType(name, np.int32, "int32")
    elif name == "int8" or name == "char":
        return DataType(name, np.int8, "int8")
    elif name == "float" or name == "float32":
        return DataType(name, np.float32, "float32")
    else:
        return None
