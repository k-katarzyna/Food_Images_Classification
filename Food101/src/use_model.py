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
    Predict the class of an image using a keras model.

    Args:
        img_filename (str): Path to the image file to be classified.
        model (tf.keras.Model): A model to use for prediction.
        label_names (list): List of label names corresponding to model output indices.
        ax (Optional[matplotlib.axes.Axes]): Matplotlib axis to plot the image with its prediction.
            If provided, the image will be displayed with the predicted label.

    Returns:
        str or None: Predicted label name as a string if `ax` is None. Otherwise, the image
            with prediction is displayed on the given axis and nothing is returned.
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
    else:
        return prediction