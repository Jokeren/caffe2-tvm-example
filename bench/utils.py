
def get_input_and_filter_shape(layout, space, channel, kernel):
    # init data
    if layout == "NCHW":
        input_shape = (1, channel, space, space)
        filter_shape = (channel, channel, kernel, kernel)
    elif layout == "NHWC":
        input_shape = (1, space, space, channel)
        filter_shape = (kernel, kernel, channel, channel)
    elif layout == "HWCN":
        input_shape = (space, space, channel, 1)
        filter_shape = (kernel, kernel, channel, channel)
    else:
        return None, None
    return input_shape, filter_shape
