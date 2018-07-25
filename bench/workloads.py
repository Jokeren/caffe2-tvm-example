class Workload(object):
    def __init__(self, space, input_channel, output_channel, kernel, pad, stride, warmup=2, run=10, depthwise=False):
        self._space = space
        self._input_channel = input_channel
        self._output_channel = output_channel
        self._kernel = kernel
        self._pad = pad
        self._stride = stride
        self._warmup = warmup
        self._run = run
        self._depthwise = depthwise

    def space(self):
        return self._space

    def input_channel(self):
        return self._input_channel

    def output_channel(self):
        return self._output_channel

    def kernel(self):
        return self._kernel

    def pad(self):
        return self._pad

    def stride(self):
        return self._stride

    def warmup(self):
        return self._warmup

    def run(self):
        return self._run

    def depthwise(self):
        return self._depthwise


def get_workloads(config):
    workloads = []
    if config == "simple_standard":
        workloads.append(Workload(56, 64, 64, 3, 0, 1))
    elif config == "caffe2_standard_oppo":
        spaces = [14, 26, 52, 104]
        channels = [64, 128, 256, 512]
        kernels = [1, 3]
        for space in spaces:
            for channel in channels:
                for kernel in kernels:
                    warmup = 1
                    run = 2
                    if space <= 26 and channel <= 64:
                        warmup = 10
                        run = 10
                    elif space >= 52 and channel >= 256:
                        warmup = 0
                        run = 1
                    workloads.append(Workload(space, channel, channel, kernel, 0, 1, warmup, run))
    elif config == "caffe2_standard":
        spaces = [14, 26, 52, 104]
        channels = [64, 128, 256, 512]
        kernels = [1, 3]
        for space in spaces:
            for channel in channels:
                for kernel in kernels:
                    warmup = 2
                    run = 10
                    if space <= 26 and channel <= 128:
                        warmup = 10
                        run = 100
                    elif space >= 52 and channel >= 256:
                        warmup = 0
                        run = 2
                    workloads.append(Workload(space, channel, channel, kernel, 0, 1, warmup, run))
    elif config == "caffe2_depthwise":
        spaces = [14, 26, 52, 104]
        channels = [64, 128, 256, 512]
        kernels = [3]
        for space in spaces:
            for channel in channels:
                for kernel in kernels:
                    warmup = 2
                    run = 10
                    if space <= 26 and channel <= 64:
                        warmup = 10
                        run = 100
                    elif space >= 52 and channel >= 256:
                        warmup = 0
                        run = 2
                    workloads.append(Workload(space, channel, channel, kernel, 0, 1, warmup, run, True))
    elif config == "mobilenet":
        # conv / s2
        workloads.append(Workload(space=224, input_channel=3, output_channel=32, pad=1, stride=2, kernel=3))
        # conv dw / s1
        workloads.append(Workload(space=112, input_channel=32, output_channel=32, pad=1, stride=1, kernel=3, warmup=10, run=100, depthwise=True))
        # conv / s1
        workloads.append(Workload(space=112, input_channel=32, output_channel=64, pad=0, stride=1, kernel=1))
        # conv dw / s2
        workloads.append(Workload(space=112, input_channel=64, output_channel=64, pad=1, stride=2, kernel=3, warmup=10, run=100, depthwise=True))
        # conv / s1
        workloads.append(Workload(space=56, input_channel=64, output_channel=128, pad=0, stride=1, kernel=1))
        # conv dw / s1
        workloads.append(Workload(space=56, input_channel=128, output_channel=128, pad=1, stride=1, kernel=3, warmup=10, run=100, depthwise=True))
        # conv / s1
        workloads.append(Workload(space=56, input_channel=128, output_channel=128, pad=0, stride=1, kernel=1))
        # conv dw / s2
        workloads.append(Workload(space=56, input_channel=128, output_channel=128, pad=1, stride=2, kernel=3, warmup=10, run=100, depthwise=True))
        # conv s1
        workloads.append(Workload(space=28, input_channel=128, output_channel=256, pad=0, stride=1, kernel=1))
        # conv dw / s1
        workloads.append(Workload(space=28, input_channel=256, output_channel=256, pad=1, stride=1, kernel=3, warmup=10, run=100, depthwise=True))
        # conv / s1
        workloads.append(Workload(space=28, input_channel=256, output_channel=256, pad=0, stride=1, kernel=1))
        # conv dw / s2
        workloads.append(Workload(space=28, input_channel=256, output_channel=256, pad=1, stride=2, kernel=3, warmup=10, run=100, depthwise=True))
        # conv / s1
        workloads.append(Workload(space=14, input_channel=256, output_channel=512, pad=0, stride=1, kernel=1))
        # 5x
        workloads.append(Workload(space=14, input_channel=512, output_channel=512, pad=1, stride=1, kernel=3, warmup=10, run=100, depthwise=True))
        workloads.append(Workload(space=14, input_channel=512, output_channel=512, pad=0, stride=1, kernel=1))

        workloads.append(Workload(space=14, input_channel=512, output_channel=512, pad=1, stride=1, kernel=3, warmup=10, run=100, depthwise=True))
        workloads.append(Workload(space=14, input_channel=512, output_channel=512, pad=0, stride=1, kernel=1))

        workloads.append(Workload(space=14, input_channel=512, output_channel=512, pad=1, stride=1, kernel=3, warmup=10, run=100, depthwise=True))
        workloads.append(Workload(space=14, input_channel=512, output_channel=512, pad=0, stride=1, kernel=1))

        workloads.append(Workload(space=14, input_channel=512, output_channel=512, pad=1, stride=1, kernel=3, warmup=10, run=100, depthwise=True))
        workloads.append(Workload(space=14, input_channel=512, output_channel=512, pad=0, stride=1, kernel=1))

        workloads.append(Workload(space=14, input_channel=512, output_channel=512, pad=1, stride=1, kernel=3, warmup=10, run=100, depthwise=True))
        workloads.append(Workload(space=14, input_channel=512, output_channel=512, pad=0, stride=1, kernel=1))
        # conv dw / s2
        workloads.append(Workload(space=14, input_channel=512, output_channel=512, pad=1, stride=2, kernel=3, warmup=10, run=100, depthwise=True))
        # conv / s1
        workloads.append(Workload(space=7, input_channel=512, output_channel=1024, pad=0, stride=1, kernel=1))
        # conv dw / s2
        workloads.append(Workload(space=7, input_channel=1024, output_channel=1024, pad=1, stride=2, kernel=3, warmup=10, run=100, depthwise=True))
        # conv / s1
        workloads.append(Workload(space=7, input_channel=1024, output_channel=1024, pad=0, stride=1, kernel=1))
    return workloads
