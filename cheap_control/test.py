from envs import *
import os
import gym
import csv
import numpy as np
import torch as th
from io import StringIO
from datetime import datetime
import json
import pickle
from dotmap import DotMap
from params import *

#env = BasePendulum.PendulumEnv(params)
env = NewPendulum.Pendulum(params)

for i_episode in range(10):
	env.set_init([np.pi+0.5, 0])
	obs = env.reset()
	done = False
	while not done:
		action = env.action_space.sample()
		action = [0]
		#print(action)
		env.render()
		obs, reward, done, info = env.step(action)

	print("Done")
env.close()
