import tensorflow as tf


rotation_layer = tf.keras.layers.RandomRotation(factor=(-0.2, 0.2))


def augment_function(image, label):

    image = rotation_layer(image)
    
    original_shape = tf.shape(image)
    batch_size = original_shape[0]
    image_height = original_shape[1]
    image_width = original_shape[2]
    image_channels = original_shape[3]

    crop_fraction = random.uniform(0.95, 1)
    crop_height = tf.cast(image_height, tf.float32) * crop_fraction
    crop_width = tf.cast(image_width, tf.float32) * crop_fraction
    crop_height = tf.cast(crop_height, tf.int32)
    crop_width = tf.cast(crop_width, tf.int32)
    image = tf.image.random_crop(image, size=[batch_size, crop_height, crop_width, image_channels])
    image = tf.image.resize(image, size=[image_height, image_width])

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=1, upper=1.1)

    return image, label


def load_and_prep_image(filename, img_shape=224):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).
    
    Args:
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224

    Returns:
    Tensor of shape (1, img_shape, img_shape, 3)
    """
    img = tf.io.read_file(filename)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize(img, [img_shape, img_shape])
    img = tf.expand_dims(img, axis=0)
    return img