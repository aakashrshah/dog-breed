import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_hub as hub
import os
import pandas as pd
import numpy as np

# Load labels and find unique breeds
labels_csv = pd.read_csv("./model/labels.csv")
unique_breeds = np.unique(labels_csv["breed"])

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
INPUT_SHAPE = [None , IMG_SIZE ,IMG_SIZE , 3 ] # Batch , height , width , color channels
OUTPUT_SHAPE = len(unique_breeds)
MODEL_URL= "https://kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow2/variations/130-224-classification/versions/1"
MODEL_PATH = './model/20240128-16181706458732-all-images-Adam.h5'


# Image processing function
def process_image(image_path, img_size=IMG_SIZE):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[img_size, img_size])
    return image

# Function to create data batches
def create_data_batches(x, batch_size=BATCH_SIZE, test_data=False):
    if test_data:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
        data_batch = data.map(process_image).batch(batch_size)
        return data_batch


# Load the model with custom objects
def load_model(model_path=MODEL_PATH):
    model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
    print("Model loaded!")
    return model


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    # Use decode_image if the format can be varying
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  # Normalize the image to [0, 1]
    return image

# Predict function
def predict(loaded_model, image_path):
    if os.path.isfile(image_path):
        # Process a single image
        image = load_and_preprocess_image(image_path)
        image = tf.expand_dims(image, 0)  # Add batch dimension
    else:
        # Process all images in the given directory
        image_paths = [os.path.join(image_path, fname) for fname in os.listdir(image_path)]
        image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        image_dataset = image_dataset.map(load_and_preprocess_image)
        image = image_dataset.batch(32)  # Batch size can be adjusted

    predictions = loaded_model.predict(image)
    # Assuming `unique_breeds` is a list of class names
    predicted_classes = [unique_breeds[np.argmax(pred)] for pred in predictions]
    
    return predicted_classes
