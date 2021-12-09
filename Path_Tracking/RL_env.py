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
  action_low: lower bound on action
  action_high: upper_bound on action
  """

  def __init__(self, dynamical_sys, reference_sys, param_dict):
      
    super(RL_env, self).__init__()

    self.param_dict = param_dict

    self.dynamical_sys = dynamical_sys
    self.reference_sys = reference_sys

    self.curr_step = 0
    self.num_steps = int(self.param_dict["total_time"] // self.param_dict["dt"])

    self.done = False

    self.cost_weights = param_dict["cost_weights"]

    self.learned = []
    self.desired = []
    self.zero = []

    self.action_space = spaces.Box(low=self.param_dict["action_low"],\
                                    high=self.param_dict["action_high"],\
                                    shape=(self.dynamical_sys.get_input_size(),),\
                                    dtype=np.float32)                        
   
    self.observation_space = spaces.Box(low=self.param_dict["action_low"],\
                                    high=self.param_dict["action_high"],\
                                 shape=(self.dynamical_sys.size() + self.reference_sys.size(),),\
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

    cost_path = self.param_dict["cost_weights"][0]*np.linalg.norm(curr_pos - reference_pos)
    cost_zero = self.param_dict["cost_weights"][1]*np.linalg.norm(zero)
    cost_input = self.param_dict["cost_weights"][2]*np.linalg.norm(action)
    total_cost = cost_path + cost_zero + cost_input 

    self.reward = -total_cost

    self.curr_step += 1

    if self.curr_step == self.num_steps:
      self.done = True

    return np.append(dstate, rstate), self.reward, self.done, {}

  def render(self, mode='console'):
    return self.learned, self.desired, self.zero

     
  def reset(self):

    dstate = self.dynamical_sys.reset()
    rstate = self.reference_sys.reset()

    self.curr_step = 0
    self.done=False
    self.learned = []
    self.desired = []
    self.zero = []

    return np.append(dstate, rstate)
    

      
