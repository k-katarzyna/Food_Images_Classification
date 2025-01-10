import os
from PIL import Image
import numpy as np


def save_dataset_as_pngs(dataset, output_dir):
    """
    Save images from tensorflow prefetch dataset as png files.

    Note: naming the class folders with sequential numbers was not a happy idea,
    because when loading such images as a dataset for the model, the numerical labels
    are given according to the order of the folder names as strings. Therefore,
    the list of full class names had to be sorted accordingly in order to correctly
    assign them to the prediction.
    """
    for i, (image, label) in enumerate(dataset):

        image = image.numpy() 
        img = Image.fromarray(image)
        class_dir = os.path.join(output_dir, str(label.numpy()))
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f"{i}.png"))