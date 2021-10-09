# -*- coding: utf-8 -*-
"""
@author: Mohsin Sarwari
Last Update: 09/18/21
"""

import numpy as np
import gym
import time
from gym import spaces
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

  def __init__(self, param_dict):

    self.param_dict = param_dict

    if self.param_dict["test"]:
      self.state = self.param_dict["initial_state"][:2]
      self.derivatives = self.param_dict["initial_state"][2:]     
    else:
      self.state = np.random.randint(low=self.param_dict["init_low"], high=self.param_dict["init_high"], size=2)
      self.derivatives = np.random.randint(low=self.param_dict["init_low"], high=self.param_dict["init_high"], size=2)

  #Size of state [x, y, x_dot, y_dot]
  def size(self):
    return 4

  def step(self, action):

    self.derivatives = np.array([action[0], self.state[0] + (self.param_dict["b"] * self.state[1])])
    self.state = self.state + (self.param_dict["dt"] * self.derivatives)

    return np.append(self.state, self.derivatives)



  def get_learned_pos(self):
    return self.state[0]

  def get_zero(self):
    return self.state[1]

  def get_input_size(self):
    return 1
     
  def reset(self):

    if self.param_dict["test"]:
      self.state = self.param_dict["initial_state"][:2]
      self.derivatives = self.param_dict["initial_state"][2:]  
    else:
      self.state = np.random.randint(low=self.param_dict["init_low"], high=self.param_dict["init_high"], size=2)
      self.derivatives = np.random.randint(low=self.param_dict["init_low"], high=self.param_dict["init_high"], size=2)

    return np.append(self.state, self.derivatives)

    

      
