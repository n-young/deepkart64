import numpy as np
import png
import ffmpeg
import os

def observe(data):
    # Make directory
    os.mkdir("./pngs")

    # Printing pngs.
    print("Printing pngs...")
    for i in range(data.shape[0]):
        d = np.reshape(data[i], (-1, data[i].shape[1] * 3))
        png.from_array(d, mode='RGB').save("./pngs/" + str(i) + ".png")
    print("Pngs printed to ./pngs!")

    # Converting pngs to mp4.
    print("Writing video...")
    ffmpeg.input("./pngs/%d.png", framerate=20).output("./video.mp4").run()
    print("Video written to ./pngs/video.mp4!")

    # Removing excess pngs.
    for i in range(data.shape[0]):
        os.remove("./pngs/" + str(i) + ".png")
    os.rmdir("./pngs")
    