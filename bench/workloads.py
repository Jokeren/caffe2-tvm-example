import tvm
import onnx
import nnvm
from nnvm import testing, graph 
from nnvm.compiler import graph_util

import numpy as np

from data_type import CompositeDataType


class ModelConfig(object):
    def __init__(self, name, input_name, input_shape, input_data_type):
        self._name = name
        self._input_name = input_name
        self._input_shape = input_shape
        self._input_data_type = input_data_type

    def name(self):
        return self._name

    def input_name(self):
        return self._input_name

    def input_shape(self):
        return self._input_shape

    def input_data_type(self):
        return self._input_data_type


class Workload(object):
    def __init__(self, model, net, params, warmup=2, run=10):
        self._model = model
        self._net = net
        self._params = params
        self._warmup = warmup
        self._run = run

    def model(self):
        return self._model

    def net(self):
        return self._net

    def params(self):
        return self._params

    def warmup(self):
        return self._warmup

    def run(self):
        return self._run


def get_input_shape(layout, space, input_channel):
    if layout == "NCHW":
        input_shape = (1, input_channel, space, space)
    elif layout == "NHWC":
        input_shape = (1, space, space, input_channel)
    elif layout == "HWCN":
        input_shape = (space, space, input_channel, 1)
    else:
        input_shape = None
    return input_shape


def init_params(net, input_name, input_shape, dtype):
    params = {}
    g = graph.create(net)
    input_shapes, _ = graph_util.infer_shape(g, data=input_shape)
    shape_dict = dict(zip(g.index.input_names, input_shapes))
    for k, v in shape_dict.items():
        if k == input_name:
            continue
        init_value = np.random.random(v).astype(dtype)
        params[k] = tvm.nd.array(init_value, ctx=tvm.cpu(0))
    return params


def create_nnvm_net(input_name, input_size, input_channel,
                    kernel_size,
                    output_channel,
                    layout="NCHW",
                    strides=(1, 1),
                    padding=(0, 0),
                    dtype=CompositeDataType.float32,
                    depthwise=False):
    net = nnvm.sym.Variable(input_name)
    groups = input_channel if depthwise else 1
    net = nnvm.sym.conv2d(net,
                          channels=output_channel,
                          kernel_size=(kernel_size, kernel_size),
                          strides=strides,
                          padding=padding,
                          use_bias=False,
                          groups=groups)

    input_shape = get_input_shape(layout, input_size, input_channel)
    params = init_params(net, input_name, input_shape, dtype=dtype.np_type())
    return net, params


def create_workload(name,
                    input_size, input_channel, output_channel, kernel_size,
                    stride=1, padding=0,
                    layout="NCHW", dtype=CompositeDataType.float32, depthwise=False,
                    warmup=2, run=10):
    input_name = "data"
    model = ModelConfig(name=name, input_name=input_name,
                        input_shape=get_input_shape(layout, input_size, input_channel),
                        input_data_type=dtype)
    net, params = create_nnvm_net(input_name, input_size, input_channel,
                                  kernel_size,
                                  output_channel,
                                  layout=layout,
                                  strides=(stride, stride),
                                  padding=(padding, padding),
                                  dtype=dtype,
                                  depthwise=depthwise)
    return Workload(model, net, params, warmup, run)


def get_model_config(model):
    if model == "squeezenetv1.1":
        return ModelConfig("squeezenetv1.1", "data_0", (1, 3, 224, 224), CompositeDataType.float32)
    elif model == "shufflenet":
        return ModelConfig("shufflenet", "gpu_0/data_0", (1, 3, 224, 224), CompositeDataType.float32)
    elif model == "personsegmentation":
        return ModelConfig("personsegmentation", "data_0", (1, 3, 192, 192), CompositeDataType.float32)
    elif model == "mobilenetv1":
        return ModelConfig("mobilenetv1", "data_0", (1, 3, 224, 224), CompositeDataType.float32)
    elif model == "resnet18v1":
        return ModelConfig("resnet18v1", "data", (1, 3, 224, 224), CompositeDataType.float32)
    else:
        return None


def get_resnet18v1_convs():
    convs = [create_workload("resnet18v1.conv1", 224, 3, 64, 7, stride=2, padding=3, run=50),
             create_workload("resnet18v1.conv2", 56, 64, 64, 3, stride=1, padding=0, run=50),
             create_workload("resnet18v1.conv3", 56, 64, 64, 3, stride=1, padding=1, run=50),
             create_workload("resnet18v1.conv4", 56, 64, 64, 3, stride=1, padding=1, run=50),
             create_workload("resnet18v1.conv5", 56, 64, 64, 3, stride=1, padding=1, run=50),
             create_workload("resnet18v1.conv6", 56, 64, 128, 1, stride=2, padding=0, run=50),
             create_workload("resnet18v1.conv7", 56, 64, 128, 3, stride=2, padding=1, run=50),
             create_workload("resnet18v1.conv8", 28, 128, 128, 3, stride=1, padding=1, run=50),
             create_workload("resnet18v1.conv9", 28, 128, 128, 3, stride=1, padding=1, run=50),
             create_workload("resnet18v1.conv10", 28, 128, 128, 3, stride=1, padding=1, run=50),
             create_workload("resnet18v1.conv11", 28, 128, 256, 1, stride=2, padding=0, run=50),
             create_workload("resnet18v1.conv12", 28, 128, 256, 3, stride=2, padding=1, run=50),
             create_workload("resnet18v1.conv13", 14, 256, 256, 3, stride=1, padding=1, run=50),
             create_workload("resnet18v1.conv14", 14, 256, 256, 3, stride=1, padding=1, run=50),
             create_workload("resnet18v1.conv15", 14, 256, 256, 3, stride=1, padding=1, run=50),
             create_workload("resnet18v1.conv16", 14, 256, 512, 1, stride=2, padding=0, run=50),
             create_workload("resnet18v1.conv17", 14, 256, 512, 3, stride=2, padding=1, run=50),
             create_workload("resnet18v1.conv18", 7, 512, 512, 3, stride=1, padding=1, run=50),
             create_workload("resnet18v1.conv19", 7, 512, 512, 3, stride=1, padding=1, run=50),
             create_workload("resnet18v1.conv20", 7, 512, 512, 3, stride=1, padding=1, run=50)]
    return convs


def get_workloads(config):
    workloads = []
    if config == "simple_standard":
        workloads.append(create_workload(config, 56, 64, 64, 3, stride=1, padding=1))
    elif config == "resnet18v1_nnvm":
        model = get_model_config("resnet18v1")
        model._input_name = "data"
        net, params = nnvm.testing.resnet.get_workload(num_layers=18, batch_size=1)
        workloads.append(Workload(model, net, params))
    elif config == "resnet18v1_onnx":
        model = get_model_config("resnet18v1")
        onnx_graph = onnx.load("onnx/" + model.name() + ".onnx")
        net, params = nnvm.frontend.from_onnx(onnx_graph)
        workloads.append(Workload(model, net, params))
    elif config == "resnet18v1_convs":
        workloads.append(get_resnet18v1_convs())
    elif config == "squeezenetv1.1_nnvm":
        model = get_model_config("squeezenetv1.1")
        model._input_name = "data"
        net, params = nnvm.testing.squeezenet.get_workload(version='1.1')
        workloads.append(Workload(model, net, params))
    elif config == "mobilenetv1_nnvm":
        model = get_model_config("mobilenetv1")
        model._input_name = "data"
        net, params = nnvm.testing.mobilenet.get_workload(batch_size=1)
        workloads.append(Workload(model, net, params))
    elif config == "personsegmentation_onnx":
        model = get_model_config("personsegmentation")
        onnx_graph = onnx.load("onnx/" + model.name() + ".onnx")
        net, params = nnvm.frontend.from_onnx(onnx_graph)
        workloads.append(Workload(model, net, params))
    return workloads
