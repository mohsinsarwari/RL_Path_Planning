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


class Reference_env():
  """
  Environment to create and track the reference path our algorithm learns to follow

  input internal_matrix defines s_dot = Mx

  input path_matrix defines r = Hx

  """

  def __init__(self, internal_matrix, path_matrix, test=False, initial_state=[0, 0, 0, 0], low=-3, high=3):

    self.low = low
    self.high = high

    self.test = test
    self.initial_state = initial_state

    self.internal_matrix = internal_matrix
    self.path_matrix = path_matrix

    self.dim = len(internal_matrix[0])

    if test:
      self.state = initial_state[:self.dim]
      self.derivatives = initial_state[self.dim:]     
    else:  
      self.state = np.random.randint(low, high=high, size=self.dim)
      self.derivatives = np.random.randint(low, high=high, size=self.dim)

  def set_dt(self, dt):
    self.dt = dt

  def size(self):
    return 2*self.dim
          
  def step(self):

    self.derivatives = np.dot(self.internal_matrix, self.state)
    self.state = self.state + (self.dt*self.derivatives)

    return np.append(self.state, self.derivatives)

  def get_reference_pos(self):
    return np.dot(self.path_matrix, self.state)
     
  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """

    if self.test:
      self.state = self.initial_state[:self.dim]
      self.derivatives = self.initial_state[self.dim:]     
    else:  
      self.state = np.random.randint(self.low, high=self.high, size=self.dim)
      self.derivatives = np.random.randint(self.low, high=self.high, size=self.dim)

    return np.append(self.state, self.derivatives)
    

      
