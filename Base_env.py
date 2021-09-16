# -*- coding: utf-8 -*-
"""


@author: Mohsin Sarwari
"""

import numpy as np
import gym
import time
from gym import spaces
from scipy import signal
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt


class Base_env():
  """
  Base environment to confirm setup works

  System:
  x_dot = u
  y_dot = x + by
  """

  def __init__(self, b, low=-3, high=3, test=False, initial_state=[0, 0, 0, 0]):

    self.low = low
    self.high = high

    self.test = test
    self.initial_state = initial_state

    if test:
      self.state = initial_state[:2]
      self.derivatives = initial_state[2:]     
    else:
      self.state = np.random.randint(low=low, high=high, size=2)
      self.derivatives = np.random.randint(low=low, high=high, size=2)

    self.action_space = spaces.Box(low=np.array([-10]),\
                                    high=np.array([10]),\
                                    dtype=np.float32)                           

    self.b = b

  def set_dt(self, dt):
    self.dt = dt

  def action_space(self):
    return self.action_space

  def observation_space(self):
    return self.observation_space

          
  def step(self, action):

    self.derivatives = np.array([action[0], self.state[0] + (self.b * self.state[1])])

    self.state = self.state + (self.dt * self.derivatives)

    return np.append(self.state, self.derivatives)


  def get_learned_pos(self):
    return self.state[0]

  def get_zero(self):
    return self.state[1]
     
  def reset(self):

    if self.test:
      self.state = initial_state[:2]
      self.derivatives = initial_state[2:]     
    else:
      self.state = np.random.randint(low=low, high=high, size=2)
      self.derivatives = np.random.randint(low=low, high=high, size=2)

    return np.append(self.state, self.derivatives)
    

      
