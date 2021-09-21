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

  def __init__(self, internal_matrix, path_matrix, dt=0.1, init_low=-3, init_high=3, test=False, initial_state=[0, 0, 0, 0]):

    self.init_low = init_low
    self.init_high = init_high

    self.test = test
    self.initial_state = initial_state

    self.internal_matrix = internal_matrix
    self.path_matrix = path_matrix
    self.dt = dt

    self.dim = len(internal_matrix[0])

    if test:
      self.state = initial_state[:self.dim]
      self.derivatives = initial_state[self.dim:]     
    else:  
      self.state = np.random.randint(low=self.init_low, high=self.init_high, size=self.dim)
      self.derivatives = np.random.randint(low=self.init_low, high=self.init_high, size=self.dim)

  def size(self):
    return 2*self.dim
          
  def step(self):

    self.derivatives = np.dot(self.internal_matrix, self.state)
    self.state = self.state + (self.dt*self.derivatives)

    return np.append(self.state, self.derivatives)

  def get_reference_pos(self):
    return np.dot(self.path_matrix, self.state)
     
  def reset(self):

    if self.test:
      self.state = self.initial_state[:self.dim]
      self.derivatives = self.initial_state[self.dim:]     
    else:  
      self.state = np.random.randint(self.low, high=self.high, size=self.dim)
      self.derivatives = np.random.randint(self.low, high=self.high, size=self.dim)

    return np.append(self.state, self.derivatives)
    

      
