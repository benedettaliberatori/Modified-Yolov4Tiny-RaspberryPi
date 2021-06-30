import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare
from yolo import Yolo
from utils import use_gpu_if_possible
import tensorflow as tf
#model = Yolo(3,3,2)
#model_dict=torch.load("model_RAdam_Augmented.pt", map_location = use_gpu_if_possible())
#model.load_state_dict(model_dict)
#dummy_input = Variable(torch.randn(1, 3, 416, 416))
#torch.onnx.export(model, dummy_input, "prova.onnx")
#model = onnx.load('prova.onnx')
#tf_rep = prepare(model)
#tf_rep.export_graph('prova.pb')

TF_PATH = "./prova.pb/saved_model.pb" # where the forzen graph is stored
TFLITE_PATH = "prova.tflite"
# protopuf needs your virtual environment to be explictly exported in the path

# make a converter object from the saved tensorflow file
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(TF_PATH,  # TensorFlow freezegraph .pb model file
                                                      input_arrays=['input'], # name of input arrays as defined in torch.onnx.export function before.
                                                      output_arrays=['output'] # name of output arrays defined in torch.onnx.export function before.
                                                      )

# tell converter which type of optimization techniques to use
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional
# converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]

converter.experimental_new_converter = True

# I had to explicitly state the ops
converter.target_spec.supported_ops = [tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.compat.v1.lite.OpsSet.SELECT_TF_OPS]

tf_lite_model = converter.convert()
# Save the model.
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)