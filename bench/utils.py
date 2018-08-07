from __future__ import print_function, absolute_import
import tvm


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
            target = opencl
            target_host = "llvm -target=%s-linux-android" % arch
            ctx = remote.cl(0)
        # TODO(Keren): how to create a mail remote?
        else:
            print("skip {0} target".format(tgt))
            return None, None, None
    else:
        if tgt == "cpu":
            target = "llvm"
        elif tgt == "gpu":
            target = "opencl"
        else:
            print("Skip %s", tgt)
            return None, None, None

        ctx = tvm.context(target, 0)
        if not ctx.exist:
            print("Skip %s", target)
            return None, None, None

        target_host = target
    return target, target_host, ctx


def get_input_and_filter_shape(
        layout,
        space,
        input_channel,
        output_channel,
        kernel,
        depthwise=False):
    if layout == "NCHW":
        input_shape = (1, input_channel, space, space)
        if depthwise:
            filter_shape = (
                input_channel,
                output_channel // input_channel,
                kernel,
                kernel)
        else:
            filter_shape = (output_channel, input_channel, kernel, kernel)
    elif layout == "NHWC":
        input_shape = (1, space, space, input_channel)
        if depthwise:
            filter_shape = (
                kernel,
                kernel,
                input_channel,
                output_channel // input_channel)
        else:
            filter_shape = (kernel, kernel, input_channel, output_channel)
    elif layout == "HWCN":
        input_shape = (space, space, input_channel, 1)
        if depthwise:
            filter_shape = (
                kernel,
                kernel,
                input_channel,
                output_channel // input_channel)
        else:
            filter_shape = (kernel, kernel, input_channel, output_channel)
    else:
        return None, None
    return input_shape, filter_shape


def get_num_ops(
        output_space,
        input_channel,
        output_channel,
        kernel,
        depthwise=False):
    if depthwise:
        return output_space * output_space * output_channel * kernel * kernel * 2
    else:
        return output_space * output_space * output_channel * \
            kernel * kernel * input_channel * 2
