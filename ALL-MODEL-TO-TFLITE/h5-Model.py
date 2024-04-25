# pip install tensorflow
import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('My_model.h5')
# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save the model to disk
with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)
