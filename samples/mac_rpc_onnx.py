"""Testcode for onnx model on a pc or a mac
"""

import os
import sys
from time import time

import numpy as np
import onnx
import tvm
from tvm.contrib import graph_runtime, util
import nnvm.frontend
import nnvm.compiler

def test_onnx_model(tgt):
    if tgt == "gpu":
        # Mobile GPU
        target = "opencl"
        ctx = remote.cl(0)
    else:
        # Mobile CPU
        target = "llvm"
        target_host = None
        ctx = remote.cpu(0)
    data_shape = ()
    onnx_graph = onnx.load("xxx.onnx")
    sym, params = nnvm.frontend.from_onnx(onnx_graph)
    opt_level = 0
    input_name = "data"

    print("Build Graph...")
    with nnvm.compiler.build_config(opt_level=opt_level, add_pass=None):
        graph, lib, params = nnvm.compiler.build(sym, target, {input_name: data_shape}, params=params)

    dtype = np.float32
    org_img = np.random(data_shape)
    img = np.asarray(org_img).astype(np.float32).copy()
    img /= 255

    # run on device
    input_data = tvm.nd.array(img.astype(dtype), ctx)
    module = graph_runtime.create(graph, rlib, ctx)
    module.set_input(input_name, input_data)
    module.set_input(**params)
    
    print('Execute')
    module.run()

    print('Benchmark')
    num_iter = 100
    ftimer = module.module.time_evaluator("run", ctx, num_iter)
    prof_res = ftimer()
    print(prof_res)

if __name__ == "__main__":
    test_onnx_model("cpu")
