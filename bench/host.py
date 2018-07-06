from __future__ import absolute_import, print_function

import time
import sys
import tvm
import topi
import numpy as np

spaces = [14, 26, 52, 104]
channels = [64, 128, 256, 512]
kernels = [1, 3]

warmup = 2
run = 10


def bench_tvm(tgt, dtype):
    if dtype == np.float32:
        tvm_dtype = "float32"
    elif dtype == np.int8:
        tvm_dtype = "int8"

    if tgt == "cpu":
        target = "llvm"
        ctx = tvm.cpu(0)
    else:
        rget = "opencl"
        ctx = tvm.context(target, 0)

    for space in spaces:
        for channel in channels:
            for kernel in kernels:
                # create schedule
                input_shape = (1, channel, space, space)
                filter_shape = (channel, channel, kernel, kernel)
                input_holder = tvm.placeholder(input_shape, dtype=tvm_dtype)
                filter_holder = tvm.placeholder(filter_shape, dtype=tvm_dtype)
                stride_holder = tvm.var("s")
                padding_holder = tvm.var("p")
                conv = topi.nn.conv2d_nchw(input_holder, filter_holder, 1, 0)
                ts = tvm.create_schedule(conv.op)

                # build code
                tl = tvm.lower(ts, [input_holder, filter_holder, stride_holder, padding_holder])
                f = tvm.build(tl, target=target, name="myconv")

                # run
                input_data = np.random.random(input_shape, )
                filter_data = np.random.random(filter_shape)
                tvm_input = tvm.nd.array(input_data.astype(dtype), ctx)
                tvm_filter = tvm.nd.array(filter_data.astype(dtype), ctx)
                start = time.time()
                for _ in range(run):
                    f(tvm_input, tvm_filter, 1, 0)
                end = time.time()
                print("input_shape: " + str(input_shape) + ", filter_shape: " + str(filter_shape) + "->" + str((end - start) / run))


def bench_caffe2(tgt, dtype):
    pass


if __name__ == "__main__":
    tgt = sys.argv[1]
    if sys.argv[2] == "float" or sys.argv[2] == "float32":
        dtype = np.float32
    elif sys.argv[2] == "int8":
        dtype = np.int8

    bench_tvm(tgt, dtype)
    bench_caffe2(tgt, dtype)
