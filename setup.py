from setuptools import setup

setup(
    name="gym_mupen64plus",
    version="0.0.3",
    install_requires=[
        "gym==0.7.4",
        "numpy==1.16.2",
        "PyYAML==5.1",
        "termcolor==1.1.0",
        "mss==4.0.2",
        "opencv-python==4.1.0.25",
        "pypng==0.0.20",
        "ffmpeg-python==0.2.0",
        "tensorflow-cpu==2.1.0",
        "dill==0.3.3",
    ],
)
