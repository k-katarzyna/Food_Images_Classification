from joblib import load
from warnings import filterwarnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from .data_preprocessing import load_and_prep_image

filterwarnings('ignore', category=UserWarning)
LABEL_NAMES = load("data/true_names.pkl")
MODEL = tf.keras.models.load_model("fine_tuned_model.keras")


def make_prediction(img_filename, model=MODEL, label_names=LABEL_NAMES, ax=None):
    """
    Predicts the class of an image using a given model. Optionally returns a matplotlib axis.

    Args:
    img_filename (str): Path to the image file.
    model (tf.keras.Model): Pre-trained model to use for prediction.
    label_names (list): List of label names corresponding to model output.
    return_ax (bool): If True, return the matplotlib axis with the image and title.
                      If False, return the prediction string.

    Returns:
    str or matplotlib.axes.Axes:
        - If `return_ax=False`: The predicted label as a string.
        - If `return_ax=True`: A matplotlib axis object with the image and title.
    """
    img = load_and_prep_image(img_filename)
    predictions = model.predict(img, verbose=0)
    prediction = label_names[int(tf.argmax(predictions, axis=1))]
    prediction = prediction.replace("_", " ").capitalize()

    if ax is not None:
        img_np = img[0].numpy().astype(np.uint8)
        ax.imshow(img_np)
        ax.axis("off")
        ax.set_title(prediction, fontsize=16)
        return None
    else:
        return prediction