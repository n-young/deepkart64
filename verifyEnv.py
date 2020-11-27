#!/bin/python
import gym, gym_mupen64plus
import numpy as np
from observe import observe

env = gym.make('Mario-Kart-Luigi-Raceway-v0')
env.reset()

data = np.array([])

for i in range(88):
    (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) # NOOP until green light
    np.append(data, env.observe())

for i in range(100):
    (obs, rew, end, info) = env.step([0, 0, 1, 0, 0]) # Drive straight
    np.append(data, env.observe())

observe(data)

raw_input("Press <enter> to exit... ")

env.close()
