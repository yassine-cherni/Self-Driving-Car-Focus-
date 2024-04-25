# pip install tensorflow onnxruntime
import onnxruntime as rt

# Load the ONNX model
onnx_model = rt.InferenceSession('My_model.onnx')
import tensorflow as tf

# Convert the ONNX model to TensorFlow
tf_rep = rt.backend.prepare(onnx_model)

# Export the TensorFlow model
tf_model_path = 'tf_model'
tf_rep.export_graph(tf_model_path)
# Load the TensorFlow model
tf_model = tf.saved_model.load(tf_model_path)
# Convert the TensorFlow model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)
