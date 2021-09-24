# -*- coding: utf-8 -*-
"""


@author: Mohsin Sarwari
"""

import numpy as np
import gym
import time
import os
from gym import spaces
from scipy import signal
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt


class RL_env(gym.Env):
  """
  Environment that mediates between dynamical system and reference generating system

  Parameters:
  dynamical_sys: system we want to learn the behavior of
  reference_sys: system that defines the reference path
  dt: time between steps
  total_time: total seconds to run simulation
  cost_weights: relative weights for different elements of cost [path_cost, zero_cost, input_cost]
  log_path: where to save info to
  """

  def __init__(self, dynamical_sys, reference_sys, dt, total_time, cost_weights, log_path):
      
    super(RL_env, self).__init__()

    self.log_path = log_path

    self.dynamical_sys = dynamical_sys
    self.reference_sys = reference_sys

    self.dt = dt
    self.curr_time = 0
    self.total_time = total_time

    self.done = False

    self.cost_weights = cost_weights

    self.learned = []
    self.desired = []
    self.zero = []

    self.total_reward = 0

    self.action_space = spaces.Box(low=-10,\
                                    high=10,\
                                    shape=(1,),\
                                    dtype=np.float32)                        
   
    self.observation_space = spaces.Box(low=-10, \
                                 high=10,\
                                 shape=(8,),\
                                 dtype=np.float32)
          
  def step(self, action):

    dstate = self.dynamical_sys.step(action)
    rstate = self.reference_sys.step()

    curr_pos = self.dynamical_sys.get_learned_pos()
    reference_pos = self.reference_sys.get_reference_pos()
    zero = self.dynamical_sys.get_zero()

    self.learned.append(curr_pos)
    self.desired.append(reference_pos)
    self.zero.append(zero)

    cost_path = self.cost_weights[0]*np.linalg.norm(curr_pos - reference_pos)
    cost_zero = self.cost_weights[1]*np.linalg.norm(zero)
    cost_input = self.cost_weights[2]*np.linalg.norm(action)
    total_cost = cost_path + cost_input + cost_zero

    self.reward = -total_cost

    self.total_reward += self.reward

    self.curr_time = np.round(self.curr_time + self.dt, 3)

    if self.curr_time <= self.total_time:
      self.done = True

    return np.append(dstate, rstate), self.reward, self.done, {}

  def get_learned_length(self):
    return len(self.learned)

  def render(self, mode='console'):
    return self.learned, self.desired, self.zero

     
  def reset(self):

    dstate = self.dynamical_sys.reset()
    rstate = self.reference_sys.reset()

    self.curr_time = 0
    self.done=False
    self.total_reward = 0
    self.learned = []
    self.desired = []
    self.zero = []

    return np.append(dstate, rstate)
    

      
