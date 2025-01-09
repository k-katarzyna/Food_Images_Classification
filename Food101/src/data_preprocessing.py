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