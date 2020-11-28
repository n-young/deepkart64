# DeepKart64

## TODO

TODO: Make the README a little less jank

## About and Getting Started

The project is split into three main parts: the setup files, the wrapper, and our source code.

The setup files (root directory) specifies instructions on how to get started with running this project. To get started, run:

```bash
docker build -t <image_name> .

docker run -it \
		--name <container_name> \
		-p 5900 \
		--mount source="<rom_path>",target=/src/gym-mupen64plus/gym_mupen64plus/ROMs,type=bind \
		<image_name>
```

We provide a Makefile to make setup, training, and visualization easy! TODO: Actually do this

Where `<rom_path>` specifies the path to your ROM file (this should be in the repo, although it probably would be best for us to remove it eventually).

The wrapper, in `./gym_mupen64plus`, is forked directly from the [gym-mupen64plus repo](https://github.com/bzier/gym-mupen64plus) with a few minor tweaks - it essentially provides a level of abstraction over the game file so we can actually interface with it like an OpenAI gym.

Our source code, in `./src`, is our own model and testing files. This is where the bulk of our project lies.

## Sources

The env wrapper was provided by the [gym-mupen64plus repo](https://github.com/bzier/gym-mupen64plus)

## Contributors

- Floria Tsui
- Kyle Reyes
- Nick Young
- Raj Paul
