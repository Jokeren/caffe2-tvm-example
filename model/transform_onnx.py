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
    if not (os.path.isfile(model_name + "_pred_net.pb") or os.path.isfile(model_name + "_init_net.pb")):
        output = open(model_name + "_pred_net.pb", "w")
        init_net_output = open(model_name + "_init_net.pb", "w")
        with open(model_name + ".onnx", 'rb') as f:
            onnx_to_caffe2(f, output, init_net_output)
