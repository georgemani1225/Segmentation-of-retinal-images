from flask import flash
import matplotlib
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# https: // stackoverflow.com/questions/53684971/assertion-failed-flask-server-stops-after-script-is-run
matplotlib.use('Agg')


def get_category(img):
    # # Read an image from a file into a numpy array
    img = mpimg.imread(img)
    # # Convert to float32
    img = tf.cast(img, tf.float32)
    # # Resize to 224x224 (size the model is expecting)
    img = tf.image.resize(img, [550, 550])
    # # Expand img dimensions from (224, 224, 3) to (1, 224, 224, 3) for set_tensor method call
    img = np.expand_dims(img, axis=0)
    tflite_model_file = 'static/model/model.tflite'
    # flash('3')
    with open(tflite_model_file, 'rb') as fid:
        tflite_model = fid.read()

    # flash('model read')

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # flash(interpreter)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    flash(input_index)
    flash(output_index)

    prediction = []
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    prediction.append(interpreter.get_tensor(output_index))

    predicted_label = np.squeeze(prediction)
    class_names = [0.419,
0.34,
0.41,
0.28,
0.27,
0.329,
0.42,
0.4,
0.33,
0.32,
0.48,
0.49,
0.588,
0.44,
0.56,
0.58,
0.51,
0.5,
0.55,
0.54,
0.66,
0.64,
0.61,
0.63,
0.589,
0.65,
0.649,
0.62,
0.59,
0.659,
0.68,
0.72,
0.71,
0.729,
0.709,
0.679,
0.7,
0.668,
0.669,
0.76,
0.81,
0.83,
0.77,
0.749,
0.74,
0.47,
0.487,
0.52,
0.442,
0.392,
0.595,
0.322,
0.475,
0.502,
0.562,
0.625,
0.698,
0.69,
0.73,
0.707,
0.612,
0.699,
0.6,
0.718,
0.75,
0.752,
0.771,
0.775,
0.744,
0.758,
0.757,
0.764,
0.742,
0.759,
0.819,
0.776,
0.79,
0.783,
0.778,
0.795,
0.805,
0.78,
0.787,
0.8,
0.825,
0.821,
0.855,
0.841,
0.84,
0.835,
0.823,
0.822,
0.842,
0.82,
0.862,
0.674,
0.67,
0.735,
0.793,
0.807,
0.827]

    flash(output_index)
