import torch
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Load the PyTorch model
pt_model = torch.load('your_model.pt')

# Convert PyTorch model to TensorFlow format
# Example using ONNX:
torch.onnx.export(pt_model, torch.randn(1, input_size), 'temp.onnx', export_params=True)

# Load the ONNX model
onnx_model = onnx.load('temp.onnx')

# Convert ONNX model to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph('temp_tf',)

# Load the TensorFlow model
tf_model = tf.saved_model.load('temp_tf')

# Optimize and quantize the TensorFlow model if needed
# (This step is optional but recommended for performance improvements)

# Convert TensorFlow model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model('temp_tf')
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)
