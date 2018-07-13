import sys

import onnx
import caffe2.python.onnx.frontend
from caffe2.python import core, utils, workspace
from caffe2.proto import caffe2_pb2

import numpy as np


#TODO(keren): uint16?
TensorNameMapper = {
        np.dtype('float32'): "GivenTensorFill",
        np.dtype('uint16'): "GivenTensorIntFill",
        np.dtype('int16'): "GivenTensorIntFill",
        np.dtype('int32'): "GivenTensorIntFill",
        np.dtype('int64'): "GivenTensorInt64Fill",
        np.dtype('uint8'): "GivenTensorStringFill",
        }


def transform_init_net(init_net):
    decompress_ops = []
    new_ops = []
    for op in init_net.op:
        if op.type == "QuantDecompZstd":
            decompress_ops.append(op)
        else:
            new_ops.append(op)

    # get decompressed results
    workspace.ResetWorkspace()
    workspace.RunNetOnce(init_net)

    for dop in decompress_ops:
        # fetch inputs
        inputs = []
        for _ in dop.input:
            inputs.append(_)
        # fetch output and construct new op
        for output in dop.output:
            blob = workspace.FetchBlob(output)
            shape = blob.shape
            values = blob
            # pass array of uint8 as a string to save storage
            # storing uint8_t has a large overhead for now
            if blob.dtype == np.dtype('uint8'):
                shape = [1]
                values = [str(blob.data)]
            op = core.CreateOperator(TensorNameMapper[blob.dtype],
                                     inputs,
                                     [output],
                                     arg=[utils.MakeArgument("shape", shape),
                                          utils.MakeArgument("values", values)])
            new_ops.append(op)

    new_init_net = caffe2_pb2.NetDef()
    new_init_net.op.extend(new_ops)
    return new_init_net


def transform_pred_net(pred_net):
    norm_output_ops = []
    new_ops = []
    for op in pred_net.op:
        # only appears in the first layer
        if op.type == "NormalizePlanarYUV":
            for output in op.output:
                for op_ in pred_net.op:
                    if output in op_.input:
                        norm_output_ops.append((op_, output))

    for op in pred_net.op:
        if op not in norm_output_ops:
            new_ops.append(op)

    # change norm_output to data
    for op, output in norm_output_ops:
        new_inputs = []
        for input_ in op.input:
            if output in input_:
                op.input.remove(output)
                break
        op.input.extend(["data"])
        new_ops.append(op)

    new_pred_net = caffe2_pb2.NetDef()
    new_pred_net.op.extend(new_ops)
    return new_pred_net

if __name__ == "__main__":
    init_net_name = sys.argv[1]
    init_net = caffe2_pb2.NetDef()
    with open(init_net_name + ".pb", 'rb') as f:
        init_net.ParseFromString(f.read())
        transform_init_net(init_net)

    pred_net_name = sys.argv[2]
    pred_net = caffe2_pb2.NetDef()
    with open(pred_net_name + ".pb", 'rb') as f:
        pred_net.ParseFromString(f.read())
        transform_pred_net(pred_net)
