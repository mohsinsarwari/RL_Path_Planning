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
      self.state = self.param_dict["initial_state_dynamic"]     
    else:
      self.state = ((self.param_dict["init_high"]-self.param_dict["init_low"])*np.random.rand(2))+self.param_dict["init_low"]

  #Size of state [x, y]
  def size(self):
    return 2

  def step(self, action):

    derivatives = np.array([action[0], self.state[0] + (self.param_dict["b"] * self.state[1])])
    self.state = self.state + (self.param_dict["dt"] * derivatives)

    return self.state

  def get_learned_pos(self):
    return self.state[0]

  def get_zero(self):
    return self.state[1]

  def get_input_size(self):
    return 1
     
  def reset(self):

    if self.param_dict["test"]:
      self.state = self.param_dict["initial_state_dynamic"] 
    else:
      self.state = ((self.param_dict["init_high"]-self.param_dict["init_low"])*np.random.rand(2))+self.param_dict["init_low"]

    return self.state

    

      
