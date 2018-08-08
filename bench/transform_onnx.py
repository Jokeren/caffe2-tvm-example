import onnx
import sys
import os
from onnx import ModelProto
from caffe2.python.onnx.backend import Caffe2Backend as c2


def onnx_to_caffe2(onnx_model, output, init_net_output):
    onnx_model_proto = ModelProto()
    onnx_model_proto.ParseFromString(onnx_model.read())

    init_net, predict_net = c2.onnx_graph_to_caffe2_net(onnx_model_proto)
    init_net_output.write(init_net.SerializeToString())
    output.write(predict_net.SerializeToString())


if __name__ == "__main__":
    model_name = sys.argv[1]
    pred_net_path = "caffe2/" + model_name + "_pred_net.pb"
    init_net_path = "caffe2/" + model_name + "_init_net.pb"
    if not (os.path.isfile(pred_net_path) or os.path.isfile(init_net_path)):
        output = open(pred_net_path, "w")
        init_net_output = open(init_net_path, "w")
        with open("onnx/" + model_name + ".onnx", 'rb') as f:
            onnx_to_caffe2(f, output, init_net_output)
