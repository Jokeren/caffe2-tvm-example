"""Testcode for onnx model on a pc or a mac
"""

import os
import sys
from time import time

import numpy as np
import onnx
import tvm
from tvm.contrib import graph_runtime, util
import nnvm
import nnvm.compiler

from models import get_model_config
from transform import transform_caffe2_to_onnx

def test_onnx_model(arch, tgt, name, opt_level):
    print("Init data...")
    model = get_model_config(name)
    input_data_shape = model.input_shape()
    input_data_type = model.input_data_type()
    org_img = np.random.random(input_data_shape)
    img = np.asarray(org_img).astype(model.input_data_type()).copy()

    print("Build graph...")
    onnx_graph = onnx.load(model.name() + ".onnx")
    sym, params = nnvm.frontend.from_onnx(onnx_graph)
    data_shapes = dict()
    data_shapes.update({k : str(v.dtype) for k, v in params.items()})
    input_name = model.input_name()
    if tgt == "cpu":
        target = "llvm"
        ctx = tvm.cpu(0)
    else:
        target = "opencl"
        ctx = tvm.context(target, 0)
    with nnvm.compiler.build_config(opt_level=opt_level, add_pass=None):
        graph, lib, params = nnvm.compiler.build(sym, target, {input_name: input_data_shape}, params=params, dtype=data_shapes)

    input_data = tvm.nd.array(img.astype(input_data_type), ctx)
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(input_name, input_data)
    module.set_input(**params)

    print('Execute...')
    module.run()

    print('Benchmark...')
    num_iter = 100
    ftimer = module.module.time_evaluator("run", ctx, num_iter)
    prof_res = ftimer()
    print(prof_res)

if __name__ == "__main__":
    arch = sys.argv[1]
    tgt = sys.argv[3]
    model_name = sys.argv[2]
    opt_level = int(sys.argv[4])
    transform_caffe2_to_onnx(model_name)
    test_onnx_model(arch, tgt, model_name, opt_level)
