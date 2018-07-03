"""Testcode for Android RPC onnx model.

To use it, start a rpc proxy with "python -m tvm.exec.rpc_proxy".
And configure the proxy host field as commented.
"""

import os
import sys
from time import time

import numpy as np
import onnx
import tvm
from tvm.contrib import graph_runtime, util, ndk, rpc
import nnvm.frontend
import nnvm.compiler

# Set to be address of tvm proxy.
proxy_host = os.environ["TVM_ANDROID_RPC_PROXY_HOST"]
proxy_port = 9090
key = "android"

# Change target configuration.
# Run `adb shell cat /proc/cpuinfo` to find the arch.
arch = "arm64"
target = "llvm -target=%s-linux-android" % arch

def test_onnx_model(tgt):
    print("Init data...")
    data_shape = (1, 3, 224, 224)
    dtype = np.float32
    org_img = np.random.random(data_shape)
    img = np.asarray(org_img).astype(np.float32).copy()

    print('RPC Connecting...')
    remote = rpc.connect(proxy_host, proxy_port, key=key)
    print('RPC Connected')

    print("Build Graph...")
    onnx_graph = onnx.load("model.onnx")
    sym, params = nnvm.frontend.from_onnx(onnx_graph)
    opt_level = 0
    input_name = "data_0"
    if tgt == "gpu":
        # Mobile GPU
        target = "opencl"
        target_host = "llvm -target=%s-linux-android" % arch
        ctx = remote.cl(0)
    else:
        # Mobile CPU
        target = "llvm -target=%s-linux-android" % arch
        target_host = None
        ctx = remote.cpu(0)
    with nnvm.compiler.build_config(opt_level=opt_level, add_pass=None):
        graph, lib, params = nnvm.compiler.build(sym, target, {input_name: data_shape}, params=params, target_host=target_host)

    print('DEPLOY: Shared Library Uploading...')
    so_name = "model-aarch64.so"
    temp = util.tempdir()
    path_so = temp.relpath(so_name)
    lib.export_library(path_so, ndk.create_shared)

    # run on remote device
    remote.upload(path_so)
    rlib = remote.load_module(so_name)
    input_data = tvm.nd.array(img.astype(dtype), ctx)
    rmodule = graph_runtime.create(graph, rlib, ctx)
    rmodule.set_input(input_name, input_data)
    rmodule.set_input(**params)
    
    print('Execute')
    rmodule.run()

    print('Benchmark')
    num_iter = 100
    ftimer = rmodule.module.time_evaluator("run", ctx, num_iter)
    prof_res = ftimer()
    print(prof_res)

if __name__ == "__main__":
    test_onnx_model("cpu")
