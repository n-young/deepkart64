import tensorflow as tf


def compress(image, n):
    """
    Compresses the given image by a factor of n.
    """
    return tf.nn.avg_pool(
        tf.cast(image, dtype=tf.float32), ksize=n, strides=n, padding="SAME"
    ), image
