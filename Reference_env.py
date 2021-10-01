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


class Reference_env():
  """
  Environment to create and track the reference path our algorithm learns to follow

  System:
  s_dot = Ms
  r = Hs

  Parameters:
  internal matrix: defines M; how the state changes
  path_matrix: defines H; how to extract the desired location from the state
  dt: time between steps
  init_low: lower bound on randomizing the state
  init_high: upper bound on randomizing the state
  test: indication if this environment is used to evaluate the model
  initial_state: fixed initial state for test environment (contains state variables then derivatives in same order)
  """

  def __init__(self, param_dict):

    self.param_dict = param_dict

    #size of s
    self.dim = len(self.param_dict["internal_matrix"][0])

    if self.param_dict["test"]:
      self.state = self.param_dict["initial_state"][:self.dim]
      self.derivatives = self.param_dict["initial_state"][self.dim:]     
    else:  
      self.state = np.random.randint(low=self.param_dict["init_low"], high=self.param_dict["init_high"], size=self.dim)
      self.derivatives = np.random.randint(low=self.param_dict["init_low"], high=self.param_dict["init_high"], size=self.dim)

  # Size of State
  def size(self):
    return 2*self.dim
          
  def step(self):

    self.derivatives = np.dot(self.param_dict["internal_matrix"], self.state)
    self.state = self.state + (self.param_dict["dt"]*self.derivatives)

    return np.append(self.state, self.derivatives)

  def get_reference_pos(self):
    return np.dot(self.param_dict["path_matrix"], self.state)
     
  def reset(self):

    if self.param_dict["test"]:
      self.state = self.param_dict["initial_state"][:self.dim]
      self.derivatives = self.param_dict["initial_state"][self.dim:]     
    else:  
      self.state = np.random.randint(low=self.param_dict["init_low"], high=self.param_dict["init_high"], size=self.dim)
      self.derivatives = np.random.randint(low=self.param_dict["init_low"], high=self.param_dict["init_high"], size=self.dim)

    return np.append(self.state, self.derivatives)
    

      
