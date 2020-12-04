import tensorflow as tf

def compress(image):
    return tf.nn.avg_pool(tf.cast(image, dtype=tf.float32), ksize=8, strides=8, padding="SAME")