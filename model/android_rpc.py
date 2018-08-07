"""
Benchmark script for Android RPC onnx model.
===========================================================

To use it in remote, start a rpc proxy with "python -m tvm.exec.rpc_tracker --port 9090"
"""
from __future__ import absolute_import, print_function

import os
import sys
from time import time

import numpy as np
import onnx
import tvm
from tvm import autotvm
from tvm import rpc
from tvm.contrib import graph_runtime, util, ndk
import nnvm.frontend
import nnvm.compiler
import contextlib


@contextlib.contextmanager
def dummy_context_mgr():
    yield None


from models import get_model_config
from transform_caffe2 import transform_caffe2_to_onnx

# Set to be address of tvm proxy.
tracker_host = os.environ["TVM_TRACKER_HOST"]
tracker_port = int(os.environ["TVM_TRACKER_PORT"])
key = "android"
log_file = "convolutions.log"


def test_onnx_model(arch, tgt, name, opt_level, key, schedule):
    print("Init data...")
    model = get_model_config(name)
    input_data_shape = model.input_shape()
    input_data_type = model.input_data_type()
    org_img = np.random.random(input_data_shape)
    img = np.asarray(org_img).astype(model.input_data_type()).copy()

    print('RPC connecting...')
    tracker = rpc.connect_tracker(tracker_host, tracker_port)
    remote = tracker.request(key, priority=0)
    print('RPC Connected')

    print("Build graph...")
    onnx_graph = onnx.load(model.name() + ".onnx")
    sym, params = nnvm.frontend.from_onnx(onnx_graph)
    data_shapes = dict()
    data_shapes.update({k: str(v.dtype) for k, v in params.items()})
    input_name = model.input_name()
    if tgt == "cpu":
        # Mobile CPU
        sch = "-device=arm_cpu" if schedule == "manual" else ""
        if arch == "armv7a":
            target = "llvm " + sch + " -target=armv7a-linux-android -mfloat-abi=soft"
        else:
            target = "llvm " + sch + " -target=%s-linux-android" % arch
        target_host = None
        ctx = remote.cpu(0)
    else:
        # Mobile GPU
        target = "opencl"
        target_host = "llvm -target=%s-linux-android" % arch
        ctx = remote.cl(0)

    log_name = "../configs/" + key + "." + model.name() + ".log"
    with autotvm.apply_history_best(log_name) if schedule == "manual" else dummy_context_mgr():
        with tvm.target.create(target):
            with nnvm.compiler.build_config(opt_level=opt_level):
                graph, lib, params = nnvm.compiler.build(
                    sym, target, {
                        input_name: input_data_shape}, params=params, target_host=target_host, dtype=data_shapes)

    print('DEPLOY: shared library uploading...')
    so_name = model.name() + "-" + arch + ".so"
    temp = util.tempdir()
    path_so = temp.relpath(so_name)
    lib.export_library(path_so, ndk.create_shared)

    # Run on remote device
    remote.upload(path_so)
    rlib = remote.load_module(so_name)
    input_data = tvm.nd.array(img.astype(input_data_type), ctx)
    rmodule = graph_runtime.create(graph, rlib, ctx)
    rmodule.set_input(input_name, input_data)
    rmodule.set_input(**params)

    print('Execute...')
    rmodule.run()

    print('Benchmark...')
    num_iter = 100
    ftimer = rmodule.module.time_evaluator("run", ctx, num_iter)
    prof_res = ftimer().mean
    print(prof_res)


if __name__ == "__main__":
    key = sys.argv[1]
    arch = sys.argv[2]
    tgt = sys.argv[3]
    model_name = sys.argv[4]
    opt_level = int(sys.argv[5])
    schedule = sys.argv[6]
    transform_caffe2_to_onnx(model_name)
    test_onnx_model(arch, tgt, model_name, opt_level, key, schedule)
