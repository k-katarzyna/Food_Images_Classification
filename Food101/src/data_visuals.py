import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def view_10_random_images(dataset, label_names, target_classes=None):
    """
    Visualize 10 random images from a TensorFlow dataset, with optional class filtering.

    :param dataset: (tf.data.Dataset) TensorFlow dataset with (image, label) tuples.
    :param label_names: (list) List of class label names.
    :param target_classes: (list, optional) A list of class names to filter images. Default is None.

    :return: None
    """

    if target_classes is None:
        max_samples = 10
    else:
        max_samples = 1500

    dataset = dataset.shuffle(buffer_size=max_samples)

    filtered_images = []

    for i, (image, label) in enumerate(dataset.take(max_samples)):
        label_name = label_names[label]
        if target_classes is None or label_name in target_classes:
            filtered_images.append((image, label))
        if len(filtered_images) >= 10:
            break

    fig, ax = plt.subplots(2, 5, figsize=(18, 8))
    ax = ax.flatten()
    if target_classes is not None and len(target_classes) == 1:
        plt.suptitle(target_classes[0].replace("_", " "))
    for i, (img, label) in enumerate(filtered_images):
        ax[i].imshow(img.numpy())
        ax[i].axis("off")
        if target_classes is None or len(target_classes) > 1:
            ax[i].set_title(label_names[label].replace("_", " "))
    plt.tight_layout()
    plt.show()


def show_saved_fig(paths, suptitle=None):
    fig, ax = plt.subplots(len(paths), 1, figsize=(14, len(paths)*5))
    if suptitle:
        plt.suptitle(suptitle)
    for i, path in enumerate(paths):
        img = mpimg.imread(path)
        ax[i].imshow(img)
        ax[i].axis("off")
    plt.tight_layout()
    plt.show()