"""Testcode for Android RPC.

To use it in remote, start a rpc proxy with "python -m tvm.exec.rpc_tracker --port 9090"
And configure the proxy host field as commented.
"""


from __future__ import absolute_import, print_function

import time
import os
import sys
import tvm
from tvm import rpc
from tvm.contrib import util, ndk, cc
import topi
from topi.util import get_const_tuple
import numpy as np

from data_type import get_data_type
from correctness import test_correctness
from workloads import get_workloads
from utils import get_input_and_filter_shape, config_arch, get_num_ops

# Set to be address of tvm proxy.
tracker_host = os.environ["TVM_TRACKER_HOST"]
tracker_port = int(os.environ["TVM_TRACKER_PORT"])
key = "android"


def build_and_run_host(phase, idx,
                       input_holder, filter_holder, kernel, space,
                       input_channel, output_channel,
                       stride, pad, layout,
                       tvm_input, tvm_filter, tvm_output,
                       ctx, ts, conv,
                       target, opt_level,
                       warmup, run):
    with tvm.build_config(opt_level=opt_level, add_pass=None):
        f_name = str(idx) + "_" + phase + "_" + str(kernel) + "_" + str(space) + "_" + str(input_channel) + "_" + str(output_channel)
        #print(tvm.lower(ts, [input_holder, filter_holder, conv], name=f_name+".S", simple_mode=False))

        # build code
        try:
            f = tvm.build(ts, [input_holder, filter_holder, conv], target, name=f_name)
        except BaseException as e:
            print("{0}--target: {1}, dtype: {2}, layout: {3}, stride: {4}, pad: {5}, input_shape: {6}, filter_shape: {7} -> failed".format( \
                  phase, target, str(tvm_input.dtype), layout, str(stride), str(pad), str(input_holder.shape), str(filter_holder.shape)))
        else:
            timer = f.time_evaluator(f.entry_name, ctx, number=run)
            cost = timer(tvm_input, tvm_filter, tvm_output).mean
            print("{0}--target: {1}, dtype: {2}, layout: {3}, stride: {4}, pad: {5}, input_shape: {6}, filter_shape: {7} -> {8}".format( \
                  phase, target, str(tvm_input.dtype), layout, str(stride), str(pad), str(input_holder.shape), str(filter_holder.shape), \
                  cost))
            if tvm_input.dtype == "float32":
                test_correctness(input_channel, output_channel, kernel, stride, pad, tvm_input.asnumpy(), \
                        tvm_filter.asnumpy(), tvm_output.asnumpy(), order=layout, depthwise=True if phase == "depthwise" else False)

            
def build_and_run_remote(phase, idx,
                         input_holder, filter_holder, kernel, space,
                         input_channel, output_channel, 
                         stride, pad, layout,
                         tvm_input, tvm_filter, tvm_output,
                         ctx, ts, conv,
                         target, target_host, opt_level,
                         warmup, run,
                         remote):
    with tvm.build_config(opt_level=opt_level, add_pass=None):
        f_name = str(idx) + "_" + phase + "_" + str(kernel) + "_" + str(space) + "_" + str(input_channel) + "_" + str(output_channel)
        #print(tvm.lower(ts, [input_holder, filter_holder, conv], name=f_name+".S", simple_mode=True))

        # build code
        # I am not sure if the configuration is runnable or not, so wrap it by a try and except
        try:
            f = tvm.build(ts, [input_holder, filter_holder, conv], target=target, target_host=target_host, name=f_name)
        except BaseException as e:
            print("{0}--target: {1}, dtype: {2}, layout: {3}, stride: {4}, pad: {5}, input_shape: {6}, filter_shape: {7} -> failed".format( \
                  phase, target, str(tvm_input.dtype), layout, str(stride), str(pad), str(input_holder.shape), str(filter_holder.shape)))
        else:
            # build code
            so_name = f_name + ".so"
            temp = util.tempdir()
            path_so = temp.relpath(so_name)
            #f.save("libs/" + so_name, "ll")
            f.export_library(path_so, ndk.create_shared)
            remote.upload(path_so)

            f = remote.load_module(so_name)
            for _ in range(warmup):
                f(tvm_input, tvm_filter, tvm_output)
            timer = f.time_evaluator(f.entry_name, ctx, number=run)
            cost = timer(tvm_input, tvm_filter, tvm_output).mean
            print("{0}--target: {1}, dtype: {2}, layout: {3}, stride: {4}, pad: {5}, input_shape: {6}, filter_shape: {7} -> {8}".format( \
                  phase, target, str(tvm_input.dtype), layout, str(stride), str(pad), str(input_holder.shape), str(filter_holder.shape), \
                  cost))
            if tvm_input.dtype == "float32":
                test_correctness(input_channel, output_channel, kernel, stride, pad, tvm_input.asnumpy(), \
                        tvm_filter.asnumpy(), tvm_output.asnumpy(), order=layout, depthwise=True if phase == "depthwise" else False)


def get_conv_ts(input_holder, filter_holder, stride, pad, layout, dtype, depthwise):
    if dtype.tvm_type() == "int8":
        output_type = "int32"
    else:
        output_type = dtype.tvm_type()
    if not depthwise:
        # s1
        conv = topi.nn.conv2d(input_holder, filter_holder, stride, pad, layout, out_dtype=output_type)
        if layout == "NCHW":
            ts = topi.generic.schedule_conv2d_nchw([conv])
        elif layout == "NHWC":
            ts = topi.generic.schedule_conv2d_nhwc([conv])
        elif layout == "HWCN":
            ts = topi.generic.schedule_conv2d_hwcn([conv])
        # s2
        #conv = topi.nn.conv2d(input_holder, filter_holder, 1, 0, layout)
        #ts = tvm.create_schedule(conv.op)
    else:
        if layout == "NCHW":
            conv = topi.nn.depthwise_conv2d_nchw(input_holder, filter_holder, [stride, stride], pad, out_dtype=output_type)
            ts = topi.generic.schedule_depthwise_conv2d_nchw([conv])
        elif layout == "NHWC":
            conv = topi.nn.depthwise_conv2d_nhwc(input_holder, filter_holder, [stride, stride], pad, out_dtype=output_type)
            ts = topi.generic.schedule_depthwise_conv2d_nhwc([conv])
        elif layout == "HWCN":
            conv = topi.nn.depthwise_conv2d_hwcn(input_holder, filter_holder, [stride, stride], pad, out_dtype=output_type)
            ts = topi.generic.schedule_depthwise_conv2d_hwcn([conv])
    return conv, ts


def bench_tvm(arch, tgt, dtype, layout, opt_level, workloads, remote):
    target, target_host, ctx = config_arch(tgt, arch, remote)
    if target is None:
        return

    for idx, workload in enumerate(workloads):
        space = workload.space()
        input_channel = workload.input_channel()
        output_channel = workload.output_channel()
        kernel = workload.kernel()
        pad = workload.pad()
        stride = workload.stride()
        warmup = workload.warmup()
        run = workload.run()
        phase = "depthwise" if workload.depthwise() else "standard"

        (input_shape, filter_shape) = get_input_and_filter_shape(layout, space, input_channel, output_channel, kernel, workload.depthwise())
        input_holder = tvm.placeholder(input_shape, dtype=dtype.tvm_type())
        filter_holder = tvm.placeholder(filter_shape, dtype=dtype.tvm_type())
        stride_holder = tvm.var("s")
        padding_holder = tvm.var("p")

        input_data = np.random.random(input_shape)
        filter_data = np.random.random(filter_shape)

        # create schedule
        with tvm.target.create(target):
            try:
                conv, ts = get_conv_ts(input_holder, filter_holder, stride, pad, layout, dtype, workload.depthwise())
            except BaseException as e:
                print("standard--target: {0}, dtype: {1}, layout: {2}, input_shape: {3}, filter_shape: {4} -> schedule skip".format( \
                      target, str(input_holder.dtype), layout, str(input_holder.shape), str(filter_holder.shape)))
                continue
            else:
                try:
                    tvm_input = tvm.nd.array(input_data.astype(dtype.np_type()), ctx)
                    tvm_filter = tvm.nd.array(filter_data.astype(dtype.np_type()), ctx)
                    tvm_output = tvm.nd.array(np.zeros(get_const_tuple(conv.shape), dtype=conv.dtype), ctx)
                    if layout == "NCHW":
                        output_space = get_const_tuple(conv.shape)[2]
                    elif layout == "NHWC":
                        output_space = get_const_tuple(conv.shape)[1]
                    elif layout == "HWCN":
                        output_space = get_const_tuple(conv.shape)[0]
                    print("ops: {0}".format(get_num_ops(output_space, input_channel, output_channel, kernel, workload.depthwise())))

                    if remote is None:
                        build_and_run_host(phase, idx,
                                           input_holder, filter_holder,
                                           kernel, space,
                                           input_channel, output_channel,
                                           stride, pad, layout,
                                           tvm_input, tvm_filter, tvm_output,
                                           ctx, ts, conv,
                                           target, opt_level,
                                           warmup, run)
                    else:
                        build_and_run_remote(phase, idx,
                                             input_holder, filter_holder,
                                             kernel, space,
                                             input_channel, output_channel,
                                             stride, pad, layout,
                                             tvm_input, tvm_filter, tvm_output,
                                             ctx, ts, conv,
                                             target, target_host, opt_level,
                                             warmup, run,
                                             remote)
                except BaseException as e:
                    print(e)
                    print("{0}--target: {1}, dtype: {2}, layout: {3}, input_shape: {4}, filter_shape: {5} -> run skip".format( \
                          phase, target, str(input_holder.dtype), layout, str(input_holder.shape), str(filter_holder.shape)))
                else:
                    continue


if __name__ == "__main__":
    arch = sys.argv[1]
    if sys.argv[2] == "remote":
        # connect to the proxy
        tracker = rpc.connect_tracker(tracker_host, tracker_port)
        remote = tracker.request(key)
    else:
        remote = None

    if len(sys.argv) > 3:
        target = sys.argv[3]
        dtype = get_data_type(sys.argv[4])
        layout = sys.argv[5]
        opt_level = int(sys.argv[6])
        workloads = get_workloads(sys.argv[7])

        bench_tvm(arch, target, dtype, layout, opt_level, workloads, remote)
    else:
        for target in ["cpu"]:
            for dtype in ["int8", "float"]:
                for layout in ["NCHW", "NHWC", "HWCN"]:
                    for opt_level in [3]:
                        for workloads in ["caffe2_depthwise", "caffe2_standard", "mobilenet"]:
                            bench_tvm(arch, target, get_data_type(dtype), layout, opt_level, get_workloads(workloads), remote)
