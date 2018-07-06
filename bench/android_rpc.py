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

# conv configs
spaces = [14]
channels = [64]
kernels = [3]

warmup = 2
run = 10

# Set to be address of tvm proxy.
proxy_host = os.environ["TVM_ANDROID_RPC_PROXY_HOST"]
proxy_port = 9090
key = "android"

# Change target configuration.
# Run `adb shell cat /proc/cpuinfo` to find the arch.
arch = "arm64"
target = "llvm -target=%s-linux-android" % arch

def bench_tvm(tgt, dtype, layout, opt_level):
    # connect to the proxy
    remote = rpc.connect(proxy_host, proxy_port, key=key)

    if tgt == "cpu":
        # Mobile CPU
        target = "llvm -target=%s-linux-android" % arch
        target_host = None
        ctx = remote.cpu(0)
    else:
        # Mobile GPU
        target = "opencl"
        target_host = "llvm -target=%s-linux-android" % arch
        ctx = remote.cl(0)

    print("------------------")
    print("standard\n")

    for space in spaces:
        for channel in channels:
            for kernel in kernels:
                # init data
                if layout == "NCHW":
                    input_shape = (1, channel, space, space)
                    filter_shape = (channel, channel, kernel, kernel)
                else:
                    continue

                input_data = np.random.random(input_shape)
                filter_data = np.random.random(filter_shape)
                input_holder = tvm.placeholder(input_shape, dtype=dtype.tvm_type())
                filter_holder = tvm.placeholder(filter_shape, dtype=dtype.tvm_type())
                stride_holder = tvm.var("s")
                padding_holder = tvm.var("p")
                tvm_input = tvm.nd.array(input_data.astype(dtype.np_type()), ctx)
                tvm_filter = tvm.nd.array(filter_data.astype(dtype.np_type()), ctx)

                # create schedule
                with tvm.target.create(target):
                    conv = topi.nn.conv2d(input_holder, filter_holder, 1, 0, layout)
                    if layout == "NCHW":
                        ts = topi.generic.schedule_conv2d_nchw([conv])
                    else:
                        continue

                tvm_output = tvm.nd.array(np.zeros(get_const_tuple(conv.shape), dtype=conv.dtype), ctx)

                with tvm.build_config(opt_level=opt_level, add_pass=None):
                    f_name = "myconv_" + str(kernel) + "_" + str(space) + "_" + str(channel)

                    # build code
                    try:
                        f = tvm.build(ts, [input_holder, filter_holder, conv], target=target, target_host=target_host, name=f_name)
                    except BaseException:
                        print(f_name + " " + str(layout) + " " + target + " error!")
                    else:
                        # build code
                        so_name = f_name + "-arch64.so"
                        temp = util.tempdir()
                        path_so = temp.relpath(so_name)
                        print(so_name)
                        f.export_library(path_so, ndk.create_shared)
                        remote.upload(path_so)

                        f = remote.load_module(so_name)
                        start = time.time()
                        for _ in range(warmup):
                            f(tvm_input, tvm_filter, tvm_output)
                        for _ in range(run):
                            f(tvm_input, tvm_filter, tvm_output)
                        end = time.time()
                        print("input_shape: " + str(input_shape) + ", filter_shape: " + str(filter_shape) + "->" + str((end - start) / run))

    #print("depthwise")
    #print("------------------")

    #for space in spaces:
    #    for channel in channels:
    #        for kernel in kernels:
    #            # create schedule
    #            input_shape = (1, channel, space, space)
    #            filter_shape = (channel, channel, kernel, kernel)
    #            input_holder = tvm.placeholder(input_shape, dtype=dtype.tvm_type())
    #            filter_holder = tvm.placeholder(filter_shape, dtype=dtype.tvm_type())
    #            stride_holder = tvm.var("s")
    #            padding_holder = tvm.var("p")
    #            conv = topi.nn.depthwise_conv2d_nchw(input_holder, filter_holder, [1, 1], 0)
    #            ts = tvm.create_schedule(conv.op)

    #            # build code
    #            tl = tvm.lower(ts, [input_holder, filter_holder, stride_holder, padding_holder])
    #            f = tvm.build(tl, target=target, target_host=target_host, name="myconvdepthwise")
    #            so_name = "myconvdepthwise" + "-" + str(space) + "-" + str(channel) + "-" + str(kernel) + "-arch64.so"
    #            temp = util.tempdir()
    #            path_so = temp.relpath(so_name)
    #            f.export_library(path_so, ndk.create_shared)
    #            remote.upload(path_so)

    #for space in spaces:
    #    for channel in channels:
    #        for kernel in kernels:
    #            # create schedule
    #            input_shape = (1, channel, space, space)
    #            filter_shape = (channel, channel, kernel, kernel)

    #            # run
    #            so_name = "myconvdepthwise" + "-" + str(space) + "-" + str(channel) + "-" + str(kernel) + "-arch64.so"
    #            f = remote.load_module(so_name)
    #            input_data = np.random.random(input_shape)
    #            filter_data = np.random.random(filter_shape)
    #            tvm_input = tvm.nd.array(input_data.astype(dtype.np_type()), ctx)
    #            tvm_filter = tvm.nd.array(filter_data.astype(dtype.np_type()), ctx)
    #            start = time.time()
    #            for _ in range(run):
    #                f(tvm_input, tvm_filter, [1, 1], 0)
    #            end = time.time()
    #            print("input_shape: " + str(input_shape) + ", filter_shape: " + str(filter_shape) + "->" + str((end - start) / run))


def bench_caffe2(tgt, dtype, layout, opt_level):
    pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        dtype = get_data_type(sys.argv[2])
        layout = sys.argv[3]
        opt_level = int(sys.argv[4])

        bench_tvm(target, dtype, layout, opt_level)
        bench_caffe2(target, dtype, layout, opt_level)
    else:
        for target in ["cpu", "gpu"]:
            for dtype in ["float", "int8"]:
                for layout in ["NCHW", "NHWC", "HWCN"]:
                    for opt_level in [1, 3]:
                        bench_tvm(target, get_data_type(dtype), layout, opt_level)
                        bench_caffe2(target, get_data_type(dtype), layout, opt_level)
