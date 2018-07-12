"""Testcode for Android RPC.

To use it, start a rpc proxy with "python -m tvm.exec.rpc_proxy".
And configure the proxy host field as commented.
"""


from __future__ import absolute_import, print_function
import time
import sys
import os
import tvm
from tvm.contrib import rpc, util, ndk
import topi
from topi.util import get_const_tuple
import numpy as np

from data_type import get_data_type
from utils import get_input_and_filter_shape

# conv configs
spaces = [14, 26, 52, 104]
channels = [64, 128, 256, 512]
standard_kernels = [1, 3]
depthwise_kernels = [3]

warmup = 2
run = 10

# Set to be address of tvm proxy.
proxy_host = os.environ["TVM_ANDROID_RPC_PROXY_HOST"]
proxy_port = 9090
key = "android"


def build_and_run(phase, input_holder, filter_holder, kernel, space, channel, 
                  tvm_input, tvm_filter, tvm_output,
                  ctx, ts, conv,
                  target, target_host, remote, layout, opt_level):
    with tvm.build_config(opt_level=opt_level, add_pass=None):
        f_name = "myconv_" + phase + "_" + str(kernel) + "_" + str(space) + "_" + str(channel)

        # build code
        # I am not sure if the configuration is runnable or not, so wrap it by a try and except
        try:
            f = tvm.build(ts, [input_holder, filter_holder, conv], target=target, target_host=target_host, name=f_name)
        except BaseException as e:
            print("{0}--target: {1}, dtype: {2}, layout: {3}, input_shape: {4}, filter_shape: {5} -> failed".format( \
                  phase, target, str(tvm_input.dtype), layout, str(input_holder.shape), str(filter_holder.shape)))
        else:
            # build code
            so_name = f_name + ".so"
            temp = util.tempdir()
            path_so = temp.relpath(so_name)
            f.export_library(path_so, ndk.create_shared)
            remote.upload(path_so)

            f = remote.load_module(so_name)
            for _ in range(warmup):
                f(tvm_input, tvm_filter, tvm_output)
            timer = f.time_evaluator(f.entry_name, ctx, number=run)
            cost = timer(tvm_input, tvm_filter, tvm_output).mean
            print("{0}--target: {1}, dtype: {2}, layout: {3}, input_shape: {4}, filter_shape: {5} -> {6}".format( \
                  phase, target, str(tvm_input.dtype), layout, str(input_holder.shape), str(filter_holder.shape), \
                  cost))


def bench_tvm(arch, tgt, dtype, layout, opt_level):
    # connect to the proxy
    remote = rpc.connect(proxy_host, proxy_port, key=key)

    if tgt == "cpu":
        # Mobile CPU
        if arch == "armv7a":
            target = "llvm -target=armv7a-linux-android -mfloat-abi=soft"
        else:
            target = "llvm -target=%s-linux-android" % arch
        target_host = None
        ctx = remote.cpu(0)
    elif tgt == "gpu":
        # Mobile GPU
        target = opencl
        target_host = "llvm -target=%s-linux-android" % arch
        ctx = remote.cl(0)
    # TODO(Keren): how to create a mail remote?
    else:
        print("skip {0} target".format(tgt))
        return

    # standard
    for space in spaces:
        for channel in channels:
            for kernel in standard_kernels:
                (input_shape, filter_shape) = get_input_and_filter_shape(layout, space, channel, kernel)
                input_holder = tvm.placeholder(input_shape, dtype=dtype.tvm_type())
                filter_holder = tvm.placeholder(filter_shape, dtype=dtype.tvm_type())
                stride_holder = tvm.var("s")
                padding_holder = tvm.var("p")

                # create optimal schedule
                with tvm.target.create(target):
                    try:
                        # s1
                        conv = topi.nn.conv2d(input_holder, filter_holder, 1, 0, layout)
                        if layout == "NCHW":
                            ts = topi.generic.schedule_conv2d_nchw([conv])
                        elif layout == "NHWC":
                            ts = topi.generic.schedule_conv2d_nhwc([conv])
                        elif layout == "HWCN":
                            ts = topi.generic.schedule_conv2d_hwcn([conv])
                        # s2
                        #conv = topi.nn.conv2d(input_holder, filter_holder, 1, 0, layout)
                        #ts = tvm.create_schedule(conv.op)
                    except BaseException:
                        print("standard--target: {0}, dtype: {1}, layout: {2}, input_shape: {3}, filter_shape: {4} -> schedule skip".format( \
                              target, str(input_holder.dtype), layout, str(input_holder.shape), str(filter_holder.shape)))
                        continue
                    else:
                        try:
                            tvm_input = tvm.nd.array(np.random.random(input_shape).astype(dtype.np_type()), ctx)
                            tvm_filter = tvm.nd.array(np.random.random(filter_shape).astype(dtype.np_type()), ctx)
                            tvm_output = tvm.nd.array(np.zeros(get_const_tuple(conv.shape), dtype=conv.dtype), ctx)

                            build_and_run("standard", input_holder, filter_holder, kernel, space, channel, 
                                          tvm_input, tvm_filter, tvm_output,
                                          ctx, ts, conv,
                                          target, target_host, remote, layout, opt_level)
                        except BaseException as e:
                            print("standard--target: {0}, dtype: {1}, layout: {2}, input_shape: {3}, filter_shape: {4} -> run skip".format( \
                                  target, str(input_holder.dtype), layout, str(input_holder.shape), str(filter_holder.shape)))
                        else:
                            continue

    # depthwise
    for space in spaces:
        for channel in channels:
            for kernel in depthwise_kernels:
                (input_shape, filter_shape) = get_input_and_filter_shape(layout, space, channel, kernel)
                input_holder = tvm.placeholder(input_shape, dtype=dtype.tvm_type())
                filter_holder = tvm.placeholder(filter_shape, dtype=dtype.tvm_type())
                stride_holder = tvm.var("s")
                padding_holder = tvm.var("p")

                # create optimal schedule
                with tvm.target.create(target):
                    try:
                        if layout == "NCHW":
                            conv = topi.nn.depthwise_conv2d_nchw(input_holder, filter_holder, [1, 1], 0)
                            ts = topi.generic.schedule_depthwise_conv2d_nchw([conv])
                        elif layout == "NHWC":
                            conv = topi.nn.depthwise_conv2d_nhwc(input_holder, filter_holder, [1, 1], 0)
                            ts = topi.generic.schedule_depthwise_conv2d_nhwc([conv])
                        elif layout == "HWCN":
                            conv = topi.nn.depthwise_conv2d_hwcn(input_holder, filter_holder, [1, 1], 0)
                            ts = topi.generic.schedule_depthwise_conv2d_hwcn([conv])
                    except BaseException:
                        print("depthwise--target: {0}, dtype: {1}, layout: {2}, input_shape: {3}, filter_shape: {4} -> schedule skip".format( \
                              target, str(input_holder.dtype), layout, str(input_holder.shape), str(filter_holder.shape)))
                        continue
                    else:
                        try:
                            tvm_input = tvm.nd.array(np.random.random(input_shape).astype(dtype.np_type()), ctx)
                            tvm_filter = tvm.nd.array(np.random.random(filter_shape).astype(dtype.np_type()), ctx)
                            tvm_output = tvm.nd.array(np.zeros(get_const_tuple(conv.shape), dtype=conv.dtype), ctx)

                            build_and_run("depthwise", input_holder, filter_holder, kernel, space, channel, 
                                          tvm_input, tvm_filter, tvm_output,
                                          ctx, ts, conv,
                                          target, target_host, remote, layout, opt_level)
                        except BaseException:
                            print("depthwise--target: {0}, dtype: {1}, layout: {2}, input_shape: {3}, filter_shape: {4} -> run skip".format( \
                                  target, str(input_holder.dtype), layout, str(input_holder.shape), str(filter_holder.shape)))
                        else:
                            continue


if __name__ == "__main__":
    arch = sys.argv[1]
    if len(sys.argv) > 2:
        target = sys.argv[2]
        dtype = get_data_type(sys.argv[3])
        layout = sys.argv[4]
        opt_level = int(sys.argv[5])

        bench_tvm(target, dtype, layout, opt_level)
    else:
        for target in ["cpu", "opencl"]:
            for dtype in ["int8", "float"]:
                for layout in ["NCHW", "NHWC", "HWCN"]:
                    for opt_level in [3]:
                        bench_tvm(arch, target, get_data_type(dtype), layout, opt_level)
