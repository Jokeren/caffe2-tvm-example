from __future__ import absolute_import
from caffe2.python import workspace, brew
from caffe2.python.model_helper import ModelHelper
import numpy as np


def test_correctness(
        input_channel,
        output_channel,
        kernel,
        stride,
        pad,
        tvm_input,
        tvm_filter,
        tvm_output,
        dtype="float32",
        order="NCHW",
        depthwise=False):
    if dtype is not "float32":
        return

    model = ModelHelper(name="testnet")
    if depthwise:
        if order != "NCHW":
            return
        brew.group_conv(
            model,
            'data',
            "conv",
            input_channel,
            output_channel,
            weight_init=('GivenTensorFill', {'values': tvm_filter}),
            kernel=kernel,
            stride=stride,
            pad=pad,
            group=input_channel,
            order=order,
        )
    else:
        if order != "NCHW" and order != "NHWC":
            return
        brew.conv(
            model,
            'data',
            "conv",
            input_channel,
            output_channel,
            weight_init=('GivenTensorFill', {'values': tvm_filter}),
            kernel=kernel,
            stride=stride,
            pad=pad,
            order=order)
    workspace.ResetWorkspace()
    workspace.RunNetOnce(model.param_init_net)
    workspace.FeedBlob("data", tvm_input)
    workspace.CreateNet(model.net)
    workspace.RunNet(model.net)
    caffe2_output = workspace.FetchBlob("conv")
    #print "tvm"
    #print tvm_output.shape
    #print "caffe2"
    #print caffe2_output.shape
    np.assert_allclose(caffe2_output, tvm_output)
