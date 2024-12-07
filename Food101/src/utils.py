import os
from PIL import Image
import numpy as np


def save_dataset_as_pngs(dataset, output_dir):
    """Save images from tensorflow prefetch dataset as png files."""
    for i, (image, label) in enumerate(dataset):

        image = image.numpy() 
        img = Image.fromarray(image)
        class_dir = os.path.join(output_dir, str(label.numpy()))
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f"{i}.png"))