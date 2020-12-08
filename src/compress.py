import tensorflow as tf


def compress(image):
    """
    Compresses the given image by a factor of 4.
    """
    return tf.nn.avg_pool(
        tf.cast(image, dtype=tf.float32), ksize=10, strides=10, padding="SAME"
    ), image
