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


class RL_env(gym.Env):
  """
  Environment that mediates between dynamical system and reference generating system
  """

  def __init__(self, dynamical_sys, reference_sys, total_time, dt, cost_weights, folder):
      
    super(RL_env, self).__init__()

    self.folder = folder

    self.dynamical_sys = dynamical_sys
    self.dynamical_sys.set_dt(dt)

    self.reference_sys = reference_sys
    self.reference_sys.set_dt(dt)

    self.curr_time = 0
    self.dt = dt
    self.total_time = total_time

    self.done = False

    self.times = np.arange(0, self.total_time+self.dt, self.dt)

    self.cost_weights = cost_weights

    self.learned = []
    self.desired = []
    self.zero = []

    self.action_space = dynamical_sys.action_space()                          
   
    self.observation_space = self.observation_space = spaces.Box(low=-10, \
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

    self.curr_time = self.curr_time + self.dt

    if self.curr_time >= self.total_time:
      self.done = True

    return np.append(dstate, rstate), self.reward, self.done, {}

  def get_learned_length(self):
    return len(self.learned)

  def render(self, mode='console'):

    # plt.figure(2)
    # plt.plot(self.times, self.learned, label = "learned")
    # plt.plot(self.times, self.desired, label = "desired")
    # plt.xlabel('time')
    # # Set the y axis label of the current axis.
    # plt.ylabel('position')
    # # Set a title of the current axes.
    # plt.title('Learned vs Desired')
    # # show a legend on the plot
    # plt.legend()
    # # Display a figure.
    # plt.savefig(self.folder + "/path.png")

    # plt.figure(3)
    # plt.plot(self.times, self.zero)    
    # plt.xlabel('time')
    # # Set the y axis label of the current axis.
    # plt.ylabel('y')
    # # Set a title of the current axes.
    # plt.title('Zero Dynamics')
    # # Display a figure.
    # plt.savefig(self.folder + "/zero.png")


    # plt.figure(2)
    # plt.plot(self.times, self.costs, label="total cost")
    # plt.plot(self.times, self.costs_path, label="path cost") 
    # plt.plot(self.times, self.costs_zero, label="zero cost") 
    # plt.plot(self.times, self.costs_input, label="input cost")     
    # plt.xlabel('time')
    # # Set the y axis label of the current axis.
    # plt.ylabel('cost')
    # # Set a title of the current axes.
    # plt.title('Costs')
    # # show a legend on the plot
    # plt.legend()
    # # Display a figure.
    # plt.savefig(self.folder + "/cost.png")

    return self.times, self.learned, self.desired, self.zero

     
  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    dstate = self.dynamical_sys.reset()
    rstate = self.reference_sys.reset()

    self.curr_time = 0
    self.done=False
    self.learned = []
    self.desired = []
    self.zero = []
    self.costs = []
    self.costs_path = []
    self.costs_input = []
    self.costs_zero = [] 


    return np.append(dstate, rstate)
    

      
