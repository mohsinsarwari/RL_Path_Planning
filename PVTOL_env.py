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
  x_ddot = -sin(theta)u_1 + eps cos(theta)u_2
  y_ddot = cos(theta)u_1 + eps sin(theta)u_2 - 1
  theta_ddot = u2

  State:
  [x, y, theta, x_dot, y_dot, theta_dot]

  Parameters:
  eps: coupling between rolling and lateral movement
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
      self.state = np.random.randint(low=self.param_dict["init_low"], high=self.param_dict["init_high"], size=6)

  #Size of state [x, y, theta, x_dot, y_dot, theta_dot]
  def size(self):
    return 6

  def step(self, action):
    
    u1 = action[0]
    u2 = action[1]
    x = self.state[0]
    y = self.state[1]
    theta = self.state[2]
    x_dot = self.state[3]
    y_dot = self.state[4]
    theta_dot = self.state[5]

    derivatives = np.array([x_dot, y_dot, theta_dot, 
                            (-np.sin(theta)*u1) + (eps*np.cos(theta)*u2),
                            (np.cos(theta)*u1) + (eps*np.sin(theta)*u2) -1,
                            u2])

    self.state = self.state + (self.param_dict["dt"] * derivatives)

    return self.state

  def get_learned_pos(self):
    return (self.state[0], self.state[1])

  def get_zero(self):
    return self.state[2]

  def get_input_size(self):
    return 2
     
  def reset(self):

    if self.param_dict["test"]:
      self.state = self.param_dict["initial_state_dynamic"] 
    else:
      self.state = np.random.randint(low=self.param_dict["init_low"], high=self.param_dict["init_high"], size=6)

    return self.state

    

      
