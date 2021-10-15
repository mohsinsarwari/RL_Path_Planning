# -*- coding: utf-8 -*-
"""


@author: Mohsin Sarwari
"""

import numpy as np
import torch as th
import gym
import time
import os
from gym import spaces
from scipy import signal
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt


class Basic_env(gym.Env):
  """
  Environment that mediates between dynamical system and reference generating system

  Parameters:
  dynamical_sys: system we want to learn the behavior of
  reference_sys: system that defines the reference path
  dt: time between steps
  total_time: total seconds to run simulation
  cost_weights: relative weights for different elements of cost [path_cost, zero_cost, input_cost]
  log_path: where to save info to
  action_low: lower bound on action
  action_high: upper_bound on action
  """

  def __init__(self, log_path):
      
    super(Basic_env, self).__init__()


    param_dict = {
        #shared params
        'dt': 0.1,
        'init_low': -10,
        'init_high': 10,
        'test': False,
        #RL_env parameters
        'total_time': 10,
        'total_timesteps': 1000000,
        'cost_weights': [1, 0.2, 0],
        'test_sizes': [0.2, 1, 3],
        #base env parameters
        'b' : -2,
        'action_high': 10,
        'action_low': -10,
        #reference env parameters
        'internal_matrix': [[0, -1], [1, 0]],
        'path_matrix': [0, 1],
        #model parameters
        'policy_kwarg': dict(activation_fn=th.nn.Tanh),
        'eval_freq': 50000,
        'save_freq': 10000,
        'gamma': 0.98,
    }


    self.param_dict = param_dict

    self.dt = param_dict["dt"]

    self.log_path = log_path

    self.curr_step = 0
    self.num_steps = int(self.param_dict["total_time"] // self.param_dict["dt"])

    self.done = False

    self.cost_weights = param_dict["cost_weights"]

    self.learned = []
    self.desired = []
    self.zero = []

    self.b = -2

    self.action_space = spaces.Box(low=self.param_dict["action_low"],\
                                    high=self.param_dict["action_high"],\
                                    shape=(1,),\
                                    dtype=np.float32)                        
   
    self.observation_space = spaces.Box(low=self.param_dict["action_low"],\
                                    high=self.param_dict["action_high"],\
                                 shape=(4,),\
                                 dtype=np.float32)

    self.state = np.array([0, 0, 1, 1])
          
  def step(self, action):

    A = np.array([[0, 0, 0, 0],
                  [1, self.b, 0, 0],
                  [0, 0, 0, -1],
                  [0, 0, 1, 0]])

    C = np.array([1, 0, 0, 0])

    deriv = np.dot(A, self.state) + (action * C)

    self.state = (self.dt*deriv.T) + self.state

    cost_path = self.param_dict["cost_weights"][0]*np.linalg.norm(self.state[0] - self.state[3])
    #cost_zero = self.param_dict["cost_weights"][1]*np.linalg.norm(self.state[1])
    #cost_input = self.param_dict["cost_weights"][2]*np.linalg.norm(action)
    total_cost = cost_path #+ cost_input + cost_zero

    self.reward = -total_cost

    self.curr_step += 1

    if self.curr_step == self.num_steps:
      self.done = True

    return self.state, self.reward, self.done, {}

  def render(self, mode='console'):
    return self.learned, self.desired, self.zero

     
  def reset(self):

    self.state = [0, 0, 1, 1]

    self.curr_step = 0
    self.done=False
    self.learned = []
    self.desired = []
    self.zero = []

    return self.state
    

      
