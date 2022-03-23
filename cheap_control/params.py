from dotmap import DotMap
from envs import *
import torch as th
import numpy as np
import datetime

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

params = DotMap()

#General Params
params.runner = "Mohsin" #just your first name
params.device = "AMS4"
params.id = 25
params.trial_id = 0 #to keep track of runs that are the same trial
params.eval_freq = 3000
params.save_freq = 100000
params.timesteps = 300000
params.gamma = 0.99
params.learning_rate = 0.0003
params.policy_kwargs = dict(activation_fn=th.nn.Tanh)
params.eps = 1
params.dt = 0.05
params.total_time = 10
params.trials = 2
params.finished = False #set to true after done running
params.algorithm = "SAC"
params.use_sde = True

#Env Specific Params
params.envs.manipulator.run = True
params.envs.pendulum.run = False
params.envs.pvtol.run = False
params.envs.cartpole.run = False
params.envs.basependulum.run = False
params.envs.baseenv.run = False
params.envs.quadrotor.run = False #not ready

params.envs.manipulator.env = Manipulator.Manipulator
params.envs.manipulator.eval_env = Manipulator.Manipulator
params.envs.manipulator.k1 = 1
params.envs.manipulator.k2 = 5
params.envs.manipulator.k3 = 1
params.envs.manipulator.b1 = 0.5
params.envs.manipulator.b2 = 0.1
params.envs.manipulator.max_input = 4
params.envs.manipulator.min_input = -4
params.envs.manipulator.init_low = [-0.3, -1.5, -0.1, -0.1] # note: the randomization of theta is around the value chosen for phi
params.envs.manipulator.init_high = [0.3, 1.5, 0.1, 0.1]
params.envs.manipulator.integration = "direct" # direct or sequential
params.envs.manipulator.alpha = 1 #scale infront of theta phi dot term
params.envs.manipulator.cost_func = 3 # 1/3 is: theta 2/4 is phi

params.envs.basependulum.env = BasePendulum.PendulumEnv #state = [th, th_dot], but obs = [sin(th), cos(th), th_dot]
params.envs.basependulum.eval_env = BasePendulum.PendulumEnv #extra env for eval callback

params.envs.baseenv.env = BaseEnv.BaseEnv #state = [th, th_dot], but obs = [sin(th), cos(th), th_dot]
params.envs.baseenv.eval_env = BaseEnv.BaseEnv #extra env for eval callback
params.envs.baseenv.max_input = 1
params.envs.baseenv.min_input = -1
params.envs.baseenv.init_low = -1
params.envs.baseenv.init_high = 1

params.envs.pendulum.env = Pendulum.Pendulum #state = [th, th_dot], but obs = [sin(th), cos(th), th_dot]
params.envs.pendulum.eval_env = Pendulum.Pendulum #extra env for eval callback
params.envs.pendulum.m = 1 #mass of pendulum
params.envs.pendulum.l = 1 #half the length of pendulum (length to com)
params.envs.pendulum.g = 10 #gravity
params.envs.pendulum.max_input = 4
params.envs.pendulum.min_input = -4
params.envs.pendulum.init_low = [-np.pi, -1] #state, not obs
params.envs.pendulum.init_high = [np.pi, 1]
params.envs.pendulum.alpha = 0.25 #scale infront of theta phi dot term
params.envs.pendulum.cost_func = 2 # 1 is: theta^2 + eps*u^2; 2 is: theta^2 + sqrt(eps)*theta_dot^2 + eps*u^2

params.envs.pvtol.env = Pvtol.Pvtol
params.envs.pvtol.eval_env = Pvtol.Pvtol
params.envs.pvtol.k = 0.01
params.envs.pvtol.m = 1
params.envs.pvtol.g = 1
params.envs.pvtol.thresh = 10 #state threshold
params.envs.pvtol.max_input = np.array([5, 2])
params.envs.pvtol.min_input = np.array([0, -2])
params.envs.pvtol.init_low = [-2, -2, 0, 0, 0, 0]
params.envs.pvtol.init_high = [2, 2, 0, 0, 0, 0]
params.envs.pvtol.cost_func = 1

params.envs.quadrotor.env = Quadrotor.Quadrotor
params.envs.quadrotor.eval_env = Quadrotor.Quadrotor
params.envs.quadrotor.dt = params.dt
params.envs.quadrotor.total_time = params.total_time
params.envs.quadrotor.i_xx = 1
params.envs.quadrotor.m = 1
params.envs.quadrotor.g = 1
params.envs.quadrotor.max_input = np.array([10, 3])
params.envs.quadrotor.min_input = np.array([5, -3])
params.envs.quadrotor.init_low = -1
params.envs.quadrotor.init_high = 1

params.envs.cartpole.env = Cartpole.Cartpole
params.envs.cartpole.eval_env = Cartpole.Cartpole
params.envs.cartpole.g = 1
params.envs.cartpole.mp = 0.5
params.envs.cartpole.mc = 1
params.envs.cartpole.l = 1  # actually half the pole's length
params.envs.cartpole.thresh = 10 #state threshold
params.envs.cartpole.lam = 0.01
params.envs.cartpole.max_input = 5
params.envs.cartpole.min_input = -5
params.envs.cartpole.init_low = [0, -np.pi, 0, 0]
params.envs.cartpole.init_high = [0, np.pi, 0, 0]
params.envs.cartpole.cost_func = 1
