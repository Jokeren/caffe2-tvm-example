"""
Benchmarking convolution performance for Android RPC.
===========================================================

To use it in remote, start a rpc proxy with "python -m tvm.exec.rpc_tracker --port 9090"
"""


from __future__ import absolute_import, print_function

import contextlib
import os
import sys

import nnvm
import numpy as np
import tvm
from tvm import rpc, autotvm
from tvm.contrib import util, ndk, graph_runtime
from workloads import get_workloads

# Change logging level to DEBUG to dump TVM IR
import logging
logging.basicConfig(level=logging.INFO)

# Set it to be address of tvm proxy.
tracker_host = os.environ["TVM_TRACKER_HOST"]
tracker_port = int(os.environ["TVM_TRACKER_PORT"])
key = "android"


@contextlib.contextmanager
def dummy_context_mgr():
    yield None


def run_tvm(name, input_name, input_data,
            graph, lib, params,
            warmup, run, remote,
            ctx):
    if remote is not None:
        logging.debug('Shared library uploading...')
        so_name = name + ".so"
        temp = util.tempdir()
        path_so = temp.relpath(so_name)
        lib.export_library(path_so, ndk.create_shared)
        remote.upload(path_so)
        rlib = remote.load_module(so_name)
        module = graph_runtime.create(graph, rlib, ctx)
    else:
        module = graph_runtime.create(graph, lib, ctx)

    module.set_input(input_name, input_data)
    module.set_input(**params)

    logging.debug('Warmup...')
    for _ in range(warmup):
        module.run()

    logging.debug('Benchmark...')
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=run)
    prof_res = np.array(ftimer().results)
    mean = np.mean(prof_res)
    stddev = np.std(prof_res)
    logging.info(name + "->running time: " + str(mean) + ", stddev: " + str(stddev))


def config_arch(tgt, arch, schedule, remote=None):
    if remote is not None:
        if tgt == "cpu":
            # Mobile CPU
            sch = "-device=arm_cpu" if schedule == "manual" else ""
            if arch == "armv7a":
                target = "llvm " + sch + " -target=armv7a-linux-android -mfloat-abi=soft"
            else:
                target = "llvm " + sch + " -target=%s-linux-android" % arch
            target_host = None
            ctx = remote.cpu(0)
        elif tgt == "gpu":
            # Mobile GPU
            target = "opencl"
            target_host = "llvm -target=%s-linux-android" % arch
            ctx = remote.cl(0)
        # TODO(Keren): how to create a mail remote?
        else:
            logging.info("Skip {0} target".format(tgt))
            return None, None, None
    else:
        if tgt == "cpu":
            target = "llvm"
        elif tgt == "gpu":
            target = "opencl"
        else:
            logging.info("Skip %s", tgt)
            return None, None, None

        ctx = tvm.context(target, 0)
        if not ctx.exist:
            logging.info("Skip %s", target)
            return None, None, None

        target_host = target
    return target, target_host, ctx


def bench_tvm(arch, tgt, workloads, remote, schedule, opt_level):
    target, target_host, ctx = config_arch(tgt, arch, schedule, remote)
    if target is None:
        return

    for workload in workloads:
        name = workload.model().name()
        input_name = workload.model().input_name()
        input_data_shape = workload.model().input_shape()
        input_data_type = workload.model().input_data_type()
        net = workload.net()
        params = workload.params()

        log_name = "../configs/" + key + ".log"
        with autotvm.apply_history_best(log_name) if schedule == "manual" else dummy_context_mgr():
            try:
                with nnvm.compiler.build_config(opt_level=opt_level):
                    data_types = dict()
                    data_types.update({k: str(v.dtype) for k, v in params.items()})
                    graph, lib, params = nnvm.compiler.build(
                        net, target, {
                            input_name: input_data_shape}, params=params, target_host=target_host, dtype=data_types)

                    img = np.asarray(np.random.random(input_data_shape)).astype(input_data_type.np_type())
                    input_data = tvm.nd.array(img, ctx)

                    warmup = workload.warmup()
                    run = workload.run()

                    run_tvm(name, input_name, input_data,
                            graph, lib, params,
                            warmup, run, remote,
                            ctx)
            except BaseException as e:
                logging.debug(e)
                logging.info("Skip " + name)
            else:
                continue


if __name__ == "__main__":
    arch = sys.argv[1]
    if sys.argv[2] == "remote":
        # Connect to the proxy
        key = sys.argv[3]
        tracker = rpc.connect_tracker(tracker_host, tracker_port)
        remote = tracker.request(key)
    else:
        remote = None

    if len(sys.argv) > 4:
        target = sys.argv[4]
        workloads = get_workloads(sys.argv[5])
        schedule = sys.argv[6]
        opt_level = int(sys.argv[7])

        bench_tvm(arch, target, workloads, remote, schedule, opt_level)
    else:
        for target in ["cpu"]:
            for workloads in ["simple_standard"]:
                for schedule in ["auto", "manual"]:
                    for opt_level in range(4):
                        bench_tvm(arch, target, get_workloads(workloads), remote, schedule, opt_level)
