"""
Autotuning convolution performance for Android RPC.
===========================================================

To use it in remote, start a rpc proxy with "python -m tvm.exec.rpc_tracker --port 9090"
"""


from __future__ import division
from topi.util import get_const_tuple
import tvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm import autotvm, rpc
from tvm.contrib.util import tempdir

import numpy as np
import fast_winograd
import collections
import os

# import logging
# logging.basicConfig(level=logging.DEBUG)

######################################################################
# Configurations
# ----------------------------------------

arch = 'armv7a'

device_key = '1plus'

task_name = 'fast_winograd'
log_file = '../configs/' + device_key + '.log'

tracker_host = '0.0.0.0'
tracker_port = 9090

tuning_option = {
    'tuner': 'xgb',
    'n_trial': 10,
    'early_stopping': 8,

    'measure_option': autotvm.measure_option(
        autotvm.use_rpc(device_key, port=tracker_port),
        number=3,
        parallel_num=1,
        timeout=100,
        build_func='ndk'),

    'use_transfer_learning': False,
}

######################################################################
# Workloads
# ----------------------------------------

Workload = collections.namedtuple("Workload", ["space", "input_channel", "output_channel", "kernel", "pad", "stride"])
WORKLOADS = []


######################################################################
# Tuning
# ----------------------------------------

def create_fast_winograd_tasks(target):
    tasks = []
    for w in WORKLOADS:
        tasks.append(autotvm.task.create(
            fast_winograd.conv2d_winograd_autotvm,
            args=(w.space, w.input_channel, w.output_channel),
            target=target))
    return tasks


def config_tasks(target):
    tasks = []
    if task_name == "fast_winograd":
        tasks += create_fast_winograd_tasks(target)
    return tasks


def tune_tasks(target):
    target = tvm.target.create(target)
    tasks = config_tasks(target)
    tuner = tuning_option['tuner']
    use_transfer_learning = tuning_option['use_transfer_learning']
    n_trial = tuning_option['n_trial']
    early_stopping = tuning_option['early_stopping']
    measure_option = tuning_option['measure_option']

    tmp_log_file = log_file + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # Do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # Pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_file)
    os.remove(tmp_log_file)


######################################################################
# Testing
# ----------------------------------------

def config_fast_winograd_funcs(ctx):
    funcs = []
    for workload in WORKLOADS:
        s, arg_bufs = fast_winograd.conv2d_winograd_autotvm(workload.space, workload.input_channel,
                                                            workload.output_channel)
        func = tvm.build(s, arg_bufs)

        np_tile = np.random.random(get_const_tuple(arg_bufs[0].shape)).astype(np.float32)
        np_u = np.random.random(get_const_tuple(arg_bufs[1].shape)).astype(np.float32)
        np_y = np.random.random(get_const_tuple(arg_bufs[2].shape)).astype(np.float32)
        tvm_tile = tvm.nd.array(np_tile, ctx)
        tvm_u = tvm.nd.array(np_u, ctx)
        tvm_y = tvm.nd.array(np_y, ctx)
        funcs.append((func, {tvm_tile, tvm_u, tvm_y}))
    return funcs


def config_funcs(task_name, ctx):
    funcs = []
    if task_name == "fast_winograd":
        funcs += config_fast_winograd_funcs(ctx)
    return funcs


def test_tasks(target):
    tracker = rpc.connect_tracker(tracker_host, tracker_port)
    remote = tracker.request(device_key)
    ctx = remote.cpu()

    with autotvm.apply_history_best(log_file):
        with tvm.target.create(target):
            print("Build funcs...")
            funcs = config_funcs(task_name, ctx)

            print("Run...")
            for i, f in enumerate(funcs):
                func = f[0]
                tensors = f[1]

                print("Export...")
                tmp = tempdir()
                if tuning_option['measure_option']['build_func'] == 'ndk':  # for android
                    from tvm.contrib import ndk
                    filename = func.entry_name + str(i) + ".so"
                    path = tmp.relpath(filename)
                    func.export_library(path, ndk.create_shared)
                else:
                    filename = func.entry_name + str(i) + ".tar"
                    path = tmp.relpath(filename)
                    func.export_library(path)

                # upload module to device
                print("Upload...")
                remote.upload(path)
                rlib = remote.load_module(filename)

                # evaluate
                print("Evaluate inference time cost...")
                ftimer = rlib.time_evaluator(rlib.entry_name, ctx, number=1, repeat=100)
                prof_res = np.array(ftimer(*tensors).results)
                print("Task %d Mean inference time (std dev): %f s (%f s)" %
                      (i, np.mean(prof_res), np.std(prof_res)))


def config_arch(arch):
    target = "llvm"
    if arch == 'armv7a':
        target = 'llvm -device=arm_cpu -target=armv7a-linux-android -mfloat-abi=soft'
    elif arch == 'rasp':
        target = "llvm -device=rasp3b"
    elif arch == 'avx2':
        target = 'llvm -mcpu=core-avx2'
    return target


if __name__ == "__main__":
    target = config_arch(arch)
    tune_tasks(target)
    test_tasks(target)
