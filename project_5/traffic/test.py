import os
import cv2
import numpy as np
import sys
import tensorflow as tf
import random

IMG_WIDTH = 30
IMG_HEIGHT = 30

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: python traffic.py model.h5 dataset_dir label")

    model = tf.keras.models.load_model(sys.argv[1])
    dataset_path = sys.argv[2]
    label = sys.argv[3]

    # Extract a random image in the label directory
    image_name = random.choice(os.listdir( os.path.join(dataset_path, label) ))
    image_data = cv2.imread(os.path.join(dataset_path, label, image_name))
    image_big = cv2.resize(image_data, (300, 300))
    image = cv2.resize(image_data, (IMG_WIDTH, IMG_HEIGHT))

    prediction_probability = model.predict(np.array([ image ]))
    predicted = np.argmax(prediction_probability) # Get the predicted label
    
    print(f"Predicted {predicted} | Real {label}")
    cv2.imshow(f"Predicted {predicted} | Real {label}", image_big)
    cv2.waitKey(0)