import os
import sys

import onnx
import caffe2.python.onnx.frontend
from caffe2.python import core, utils, workspace
from caffe2.proto import caffe2_pb2

import numpy as np
from models import get_model_config


#TODO(keren): uint16?
TensorNameMapper = {
        np.dtype('float32'): "GivenTensorFill",
        np.dtype('uint16'): "GivenTensorIntFill",
        np.dtype('int16'): "GivenTensorIntFill",
        np.dtype('int32'): "GivenTensorIntFill",
        np.dtype('int64'): "GivenTensorInt64Fill",
        np.dtype('uint8'): "GivenTensorStringFill",
        }

def copy_from(input_net):
    output_net = caffe2_pb2.NetDef()
    output_net.name = input_net.name
    if input_net.HasField("type"):
        output_net.type = input_net.type
    if input_net.HasField("device_option"):
        output_net.device_option.CopyFrom(input_net.device_option)
    output_net.arg.extend(input_net.arg)
    output_net.external_input.extend(input_net.external_input)
    output_net.external_output.extend(input_net.external_output)
    return output_net

def transform_init_net(init_net):
    decompress_ops = []
    constant_ops = []
    decompress_input_ops = []
    byte_weight_dequant_ops = []
    new_ops = []
    for op in init_net.op:
        if op.type == "QuantDecompZstd":
            decompress_ops.append(op)
            for input_ in op.input:
                for op_ in init_net.op:
                    if input_ in op_.output:
                        decompress_input_ops.append(op_)
        elif op.type == "ByteWeightDequant":
            byte_weight_dequant_ops.append(op)
        elif op.type == "ConstantFill":
            constant_ops.append(op)

    for op in init_net.op:
        if op not in decompress_ops and \
           op not in decompress_input_ops and \
           op not in byte_weight_dequant_ops and \
           op not in constant_ops:
            new_ops.append(op)

    # get decompressed results
    workspace.ResetWorkspace()
    workspace.RunNetOnce(init_net)

    # replace a byte weight dequant op with a tensor fill op
    for bop in byte_weight_dequant_ops:
        # fetch output and construct new op
        for output in bop.output:
            blob = workspace.FetchBlob(output)
            shape = blob.shape
            values = blob
            # pass array of uint8 as a string to save storage
            # storing uint8_t has a large overhead for now
            if blob.dtype == np.dtype('uint8'):
                shape = [1]
                values = [str(blob.data)]
            op = core.CreateOperator(TensorNameMapper[blob.dtype],
                                     [],
                                     [output],
                                     arg=[utils.MakeArgument("shape", shape),
                                          utils.MakeArgument("values", values)])
            new_ops.append(op)
    
    new_init_net = copy_from(init_net)
    new_init_net.op.extend(new_ops)
    return new_init_net


def transform_pred_net(pred_net):
    norm_ops = []
    norm_output_ops = []
    new_ops = []
    for op in pred_net.op:
        # only appears in the first layer
        if op.type == "NormalizePlanarYUV":
            norm_ops.append(op)
            for output in op.output:
                for op_ in pred_net.op:
                    if output in op_.input:
                        norm_output_ops.append((op_, output))

    for op in pred_net.op:
        if op not in norm_output_ops and \
           op not in norm_ops:
            new_ops.append(op)

    # change norm_output to data
    for op, output in norm_output_ops:
        new_inputs = []
        for input_ in op.input:
            if output == input_:
                new_inputs.append("data")
            else:
                new_inputs.append(input_)
        while len(op.input) > 0:
            del op.input[-1]
        op.input.extend(new_inputs)
        new_ops.append(op)

    # delete unnecessary attrs
    for op in new_ops:
        if op.type == "Conv" or op.type == "MaxPool" or op.type == "Relu":
            args = []
            for arg in op.arg:
                if arg.name != "convolution_transform_strategy" and \
                   arg.name != "shared_buffer" and \
                   arg.name != "init_params" and \
                   arg.name != "algo" and \
                   arg.name != "exhaustive_search" and \
                   arg.name != "adj" and \
                   arg.name != "hwgq":
                    args.append(arg)
            while len(op.arg) > 0:
                del op.arg[-1]
            op.arg.extend(args)

    new_pred_net = copy_from(pred_net)

    # adjust pred_net input, remove all quant inputs
    while len(new_pred_net.external_input) > 0:
        del new_pred_net.external_input[-1]
    new_inputs = []
    for input_ in pred_net.external_input:
        if "quant" not in input_:
            new_inputs.append(input_)
    new_pred_net.external_input.extend(new_inputs)
    new_pred_net.op.extend(new_ops)

    return new_pred_net


def transform_caffe2_to_onnx(model_name):
    if os.path.isfile(model_name + ".onnx"):
        return

    init_net_name = model_name + "_init_net"
    pred_net_name = model_name + "_pred_net"

    init_net = caffe2_pb2.NetDef()
    with open(init_net_name + ".pb", 'rb') as f:
        init_net.ParseFromString(f.read())
        transformed_init_net = transform_init_net(init_net)

    pred_net = caffe2_pb2.NetDef()
    with open(pred_net_name + ".pb", 'rb') as f:
        pred_net.ParseFromString(f.read())
        transformed_pred_net = transform_pred_net(pred_net)
        
    # We need to provide type and shape of the model inputs,
    model = get_model_config(model_name)
    if model.input_data_type() == np.float32:
        data_type = onnx.TensorProto.FLOAT
    data_shape = model.input_shape()
    value_info = {
            'data': (data_type, data_shape)
            }

    onnx_model = caffe2.python.onnx.frontend.caffe2_net_to_onnx_model(
            transformed_pred_net,
            transformed_init_net,
            value_info,
            )

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, model_name + ".onnx")


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
