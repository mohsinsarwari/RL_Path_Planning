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

env = Pvtol.Pvtol(params)

for i_episode in range(10):
	obs = env.reset()
	done = False
	while not done:
		action = env.action_space.sample()
		action = [1.5, 0.2]
		#print(action)
		env.render()
		obs, reward, done, info = env.step(action)

	print("Done")
env.close()
