import tensorflow as tf
import numpy as np
import png
import ffmpeg
import os


def compress(image, n):
    """
    Compresses the given image by a factor of n.
    """
    return tf.nn.avg_pool(
        tf.cast(image, dtype=tf.float32), ksize=n, strides=n, padding="SAME"
    ), image


def observe(data, filename):
    """
    Ingests a 3d numpy array, data, that contains the pixel data of a run.
    Exports it to ./src/video.mp4.
    """
    # Make directory
    print("making directory")
    os.mkdir("./pngs")

    # Printing pngs.
    print("Printing pngs...")
    for i in range(data.shape[0]):
        d = np.reshape(data[i], (-1, data[i].shape[1] * 3))
        png.from_array(d, mode="RGB").save("./pngs/" + str(i) + ".png")
    print("Pngs printed to ./pngs!")

    # Converting pngs to mp4.
    print("Writing video...")
    ffmpeg.input("./pngs/%d.png", framerate=20).output(filename).run()
    print("Video written to ./src/" + filename)

    # Removing excess pngs.
    for i in range(data.shape[0]):
        os.remove("./pngs/" + str(i) + ".png")
    os.rmdir("./pngs")
