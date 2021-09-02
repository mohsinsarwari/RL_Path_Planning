# -*- coding: utf-8 -*-
"""


@author: Michael Estrada
"""

import numpy as np
import gym
import time
from gym import spaces
from scipy import signal
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt


class Base_env(gym.Env):
  """
  Base environment to confirm setup works

  System:
  x_dot = u
  y_dot = x + by
  s_1_dot = s_2
  s_2_dot = -s_1
  """

  def __init__(self):
      
    super(Base_env, self).__init__()

    x_initial = 0
    x_dot_initial = 0
    y_initial = 0
    y_dot_initial = 0
    s_1_initial = 1
    s_1_dot_initial = 0
    s_2_initial = 0
    s_2_dot_initial = 1

    self.b = -2

    self.dt = 0.1

    self.curr_time = 0

    self.total_time = 10

    self.done = False

    self.times = np.arange(0, self.total_time+self.dt, self.dt)

    self.learned = []

    self.desired = []

    self.zero = []

    self.costs = []
    self.costs_path = []
    self.costs_input = [] 

    self.state = np.array([x_initial, x_dot_initial, \
                            y_initial, y_dot_initial, \
                            s_1_initial, s_1_dot_initial, \
                            s_2_initial, s_2_dot_initial])

    self.action_space = spaces.Box(low=np.array([-10]),\
                                    high=np.array([10]),\
                                    dtype=np.float32)                           
   
    self.observation_space = spaces.Box(low=-10, \
                                         high=10,\
                                         shape=(self.state.size,),\
                                         dtype=np.float32)
          
  def step(self, action):

    #print("action: ", str(action))
    x_dot = action[0]
    y_dot = self.state[0] + (self.b * self.state[2])
    s_1_dot = -self.state[6]
    s_2_dot = self.state[4]

    x = self.state[0] + (self.dt * self.state[1])
    y = self.state[2] + (self.dt * self.state[3])
    s_1 = self.state[4] + (self.dt * self.state[5])
    s_2 = self.state[6] + (self.dt * self.state[7])

    self.state = np.array([x, x_dot, y, y_dot, s_1, s_1_dot, s_2, s_2_dot])

    self.learned.append(x)
    self.desired.append(s_2)
    self.zero.append(y)
    self.reward = -((x - s_2)**2 + (action[0])**2 + (y)**2)
    self.rewards.append(self.reward)

    self.curr_time = self.curr_time + self.dt

    if self.curr_time >= self.total_time:
      self.done = True

    return self.state, self.reward, self.done, {}

  def render(self, mode='console'):

    plt.figure(0)
    plt.plot(self.times, self.learned, label = "learned")
    plt.plot(self.times, self.desired, label = "desired")
    plt.xlabel('time')
    # Set the y axis label of the current axis.
    plt.ylabel('position')
    # Set a title of the current axes.
    plt.title('Learned vs Desired')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.savefig("Base_Logs/path_2M.png")

    plt.figure(1)
    plt.plot(self.times, self.zero)    
    plt.xlabel('time')
    # Set the y axis label of the current axis.
    plt.ylabel('y')
    # Set a title of the current axes.
    plt.title('Zero Dynamics')
    # Display a figure.
    plt.savefig("Base_Logs/zero_2M.png")


    plt.figure(2)
    plt.plot(self.times, self.cost)    
    plt.xlabel('time')
    # Set the y axis label of the current axis.
    plt.ylabel('y')
    # Set a title of the current axes.
    plt.title('Zero Dynamics')
    # Display a figure.
    plt.savefig("Base_Logs/zero_2M.png")

     
  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    x_initial = 0
    x_dot_initial = 0
    y_initial = 0
    y_dot_initial = 0
    s_1_initial = 1
    s_1_dot_initial = 0
    s_2_initial = 0
    s_2_dot_initial = 1

    self.curr_time = 0
    self.done = False

    self.learned = []
    self.desired = []
    self.zero = []

    self.state = np.array([x_initial, x_dot_initial, \
                            y_initial, y_dot_initial, \
                            s_1_initial, s_1_dot_initial, \
                            s_2_initial, s_2_dot_initial])

    return self.state
    

      
