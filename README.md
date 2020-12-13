# DeepKart64

[Devpost Submission](https://devpost.com/software/deepkart64)

## About

This repository is for DeepKart64, our final project for CSCI 1470 - Deep Learning at Brown University. Developed by Floria Tsui, Kyle Reyes, Nick Young, and Raj Paul, DeepKart64 is a deep reinforcement learning agent that plays Mario Kart 64 on the Nintendo 64. It is an A2C model that uses convolutional layers to interpret game visuals and outputs instructions in a discrete action space. Learn more in our [Devpost Submission](https://devpost.com/software/deepkart64)!

## Getting Started

To get started, run:

```bash
docker build -t <image_name> .

docker run -it \
		--name <container_name> \
		-p 5900 \
		--mount source="<rom_path>",target=/src/gym-mupen64plus/gym_mupen64plus/ROMs,type=bind \
		<image_name>
```

Where `<rom_path>` specifies the absolute path to your ROM file. This command will build the source code and start an interactive Docker container where you can run commands. If you wish to run this in the background, run `docker run -dt` instead, and then attach to the container using `docker exec -it <container_name> /bin/bash`.

## Tweakable Parameters

Our `main.py` has a number of tweakable parameters at the top of the file. Use these to tune while training!

### Command Line Functionality

Our model also provides a number of command line options to help save and load models, as well as export video. Note that only one of the following can be used at a time.

To save a model, run `python main.py -s <model_name.pkl>`.

To load a model, run `python main.py -l <model_name.pkl>`.

To load a model then save a new one, run `python main.py -ls <to_load.pkl> <to_save.pkl>`.

To generate and export a video of a single episode, run `python main.py -o <video.mp4>`.

To load a model and then generate and export a video of a single episide, run `python main.py -lo <to_load.pkl> <video.mp4>`.

To save incremental models to a folder every `SAVE_FREQUENCY` episdoes, run `python main.py -S <archive_directory>`.

## Project Structure

The project is split into three main parts: the wrapper and our source code.

### Wrapper

The wrapper, in `./gym_mupen64plus`, is forked directly from the [gym-mupen64plus repo](https://github.com/bzier/gym-mupen64plus) with a few minor tweaks - it essentially provides a level of abstraction over the game file so we can actually interface with it like an OpenAI gym. We modified the Dockerfile to work with our code, and optimized it to load faster. We also modified some of the Mario Kart environment settings to help our agent learn faster.

### Source Code

Our source code, in `./src`, is our own model and testing files. This is where the bulk of our project lies. We have three files, `main.py`, `model.py`, and `utils.py`.

The `main.py` file is where our model is primarily run.

The `model.py` file is where our model is defined.

The `utils.py` file is where we define some useful helper functions to export videos and compress our states.

## Sources

The env wrapper was provided by the [gym-mupen64plus repo](https://github.com/bzier/gym-mupen64plus)

## Contributors

-   [Floria Tsui](https://github.com/floriatsui)
-   [Kyle Reyes](https://github.com/kylewreyes)
-   [Nick Young](https://github.com/n-young)
-   [Raj Paul](https://github.com/rpaul48)
