import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
model = tf.keras.models.load_model('.\digit.keras')
def img_to_tensor(img):
    # Resize the image to 28x28
    img = cv2.resize(img, (28, 28))

    # Normalize pixel values to the range 0-1
    img = img / 255.0

    # Reshape to (28, 28, 1)
    img = np.expand_dims(img, axis=-1)

    # Convert to TensorFlow tensor
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    # Add a batch dimension
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    
    return img_tensor
def get_prediction(image_path):
    # Load the model
    image = img_to_tensor(image_path)
    res = model.predict(image)[0]
    ans = 0
    for i in range(10):
      if res[i] > res[ans]:
        ans = i
    return ans
#img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
#print('The number is',get_prediction(img))