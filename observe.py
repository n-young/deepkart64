import numpy as np
from array2gif import write_gif

def observe(data, path="./gifs/observed.gif", fps=5):
    write_gif(data, path, fps=fps)
