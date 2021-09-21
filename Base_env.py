# -*- coding: utf-8 -*-
"""
@author: Mohsin Sarwari
Last Update: 09/18/21
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

  Parameters:
  b: "stability" of the system; negative stable, positive stable
  dt: time between steps
  init_low: lower bound on randomizing the state
  init_high: upper bound on randomizing the state
  test: indication if this environment is used to evaluate the model
  initial_state: fixed initial state for test environment (contains state variables then derivatives in same order)
  """

  def __init__(self, b=-2, dt=0.1, init_low=-3, init_high=3, test=False, initial_state=[0, 0, 0, 0]):

    self.b = b
    self.dt = dt

    self.init_low = init_low
    self.init_high = init_high

    self.test = test
    self.initial_state = initial_state

    if test:
      self.state = initial_state[:2]
      self.derivatives = initial_state[2:]     
    else:
      self.state = np.random.randint(low=self.init_low, high=self.init_high, size=2)
      self.derivatives = np.random.randint(low=self.init_low, high=self.init_high, size=2)
          
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
      self.state = self.initial_state[:2]
      self.derivatives = self.initial_state[2:]  
    else:
      self.state = np.random.randint(low=self.init_low, high=self.init_high, size=2)
      self.derivatives = np.random.randint(low=self.init_low, high=self.init_high, size=2)

    return np.append(self.state, self.derivatives)

    

      
