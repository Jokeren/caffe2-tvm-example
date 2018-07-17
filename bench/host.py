from __future__ import absolute_import, print_function

import time
import sys
import tvm
import topi
from topi.util import get_const_tuple
import numpy as np

from data_type import get_data_type

spaces = [14, 26, 52, 104]
channels = [32, 64, 128, 256, 512]
kernels = [3]

warmup = 2
run = 10


def bench_tvm(arch, tgt, dtype, layout, opt_level):
    if tgt == "cpu":
        target = "llvm"
    elif tgt == "gpu":
        target = "opencl"
    else:
        print("Skip %s", tgt)
        return

    ctx = tvm.context(target, 0)
    if not ctx.exist:
        print("Skip %s", target)
        return

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
                        f = tvm.build(ts, [input_holder, filter_holder, conv], target, name=f_name)
                    except BaseException:
                        print(f_name + " " + str(layout) + " " + target + " error!")
                    else:
                        # run
                        timer = f.time_evaluator(f.entry_name, ctx, number=run)
                        cost = timer(tvm_input, tvm_filter, tvm_output).mean
                        print("standard--target: " + target + ", layout: " + layout + ", input_shape: " + \
                              str(input_shape) + ", filter_shape: " + str(filter_shape) + "->" + str(cost))

    for space in spaces:
        for channel in channels:
            for kernel in kernels:
                # init data
                if layout == "NCHW":
                    input_shape = (1, channel, space, space)
                    filter_shape = (channel, channel, kernel, kernel)
                elif layout == "NHWC":
                    input_shape = (1, space, space, channel)
                    filter_shape = (kernel, kernel, channel, channel)
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
                    if layout == "NCHW":
                        conv = topi.nn.depthwise_conv2d_nchw(input_holder, filter_holder, 1, 0)
                        ts = topi.generic.schedule_depthwise_conv2d_nchw([conv])
                    elif layout == "NHWC":
                        conv = topi.nn.depthwise_conv2d_nhwc(input_holder, filter_holder, 1, 0)
                        ts = topi.generic.schedule_depthwise_conv2d_nhwc([conv])
                    else:
                        continue

                tvm_output = tvm.nd.array(np.zeros(get_const_tuple(conv.shape), dtype=conv.dtype), ctx)

                with tvm.build_config(opt_level=opt_level, add_pass=None):
                    f_name = "myconv_depthwise_" + str(kernel) + "_" + str(space) + "_" + str(channel)

                    # build code
                    try:
                        f = tvm.build(ts, [input_holder, filter_holder, conv], target, name=f_name)
                    except BaseException:
                        print(f_name + " error!")
                    else:
                        timer = f.time_evaluator(f.entry_name, ctx, number=run)
                        cost = timer(tvm_input, tvm_filter, tvm_output).mean
                        print("depthwise--target: " + target + ", layout: " + layout + ", input_shape: " + \
                              str(input_shape) + ", filter_shape: " + str(filter_shape) + "->" + cost)


if __name__ == "__main__":
    arch = sys.argv[1]
    if len(sys.argv) > 2:
        target = sys.argv[2]
        dtype = get_data_type(sys.argv[3])
        layout = sys.argv[4]
        opt_level = int(sys.argv[5])

        bench_tvm(arch, target, dtype, layout, opt_level)
    else:
        for target in ["cpu", "gpu"]:
            for dtype in ["float", "int8"]:
                for layout in ["NCHW", "NHWC", "HWCN"]:
                    for opt_level in [3]:
                        bench_tvm(arch, target, get_data_type(dtype), layout, opt_level)

