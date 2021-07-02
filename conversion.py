import sys
from os.path import dirname
sys.path.append("/home/jovoni/onnx-tensorflow")

import onnx
import torch
from yolo import Yolo
from utils2 import use_gpu_if_possible
from onnx_tf.backend import prepare
import tensorflow as tf
import os

if __name__ == "__main__":
    # exmample for the forward pass input
    example_input = torch.rand(1, 3, 416, 416)
    pytorch_model = Yolo(3, 3, 2)
    model_dict = torch.load("model_RAdam_Augmented.pt",
                        map_location=use_gpu_if_possible())
    pytorch_model.load_state_dict(model_dict)

    ONNX_PATH = "_model.onnx"

    torch.onnx.export(
        model=pytorch_model,
        args=example_input,
        f=ONNX_PATH,  # where should it be saved
        verbose=False,
        export_params=True,
        do_constant_folding=False,  # fold constant values for optimization
        # do_constant_folding=True,   # fold constant values for optimization
        input_names=['input'],
        output_names=['output']
    )

    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    
    TF_PATH = "./tf_model.pb"
    
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(TF_PATH)
    
    TF_LITE_PATH = "./tfl_model.pb"
    TF_PATH = "./tf_model.pb/saved_model.pb"
    
    os.environ["PATH"] = "/home/jovoni/anaconda3/"
    
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(TF_PATH,  # TensorFlow freezegraph .pb model file
                                                      input_arrays=['input'], # name of input arrays as defined in torch.onnx.export function before.
                                                      output_arrays=['output'] # name of output arrays defined in torch.onnx.export function before.
                                                      )
    
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.compat.v1.lite.OpsSet.SELECT_TF_OPS]

    tf_lite_model = converter.convert()
    # Save the model.
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tf_lite_model)
    

