#!/bin/python
import gym, gym_mupen64plus
import numpy as np
from observe import observe

# Script to verify that setup worked properly.

env = gym.make("Mario-Kart-Luigi-Raceway-v0")
env.reset()

data = []
for i in range(88):
    (obs, rew, end, info) = env.step([0, 0, 0, 0, 0])  # NOOP until green light
    data.append(obs)

for i in range(100):
    (obs, rew, end, info) = env.step([0, 0, 1, 0, 0])  # Drive straight
    data.append(obs)

# Turn on if you want a bunch of pictures, and then run `make copy` to get them locally
observe(np.array(data), "./video.mp4")

raw_input("Press <enter> to exit... ")

env.close()
