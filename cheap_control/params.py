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
params.device = "Hybrid Server"
params.id = 0
params.eval_freq = 1000
params.save_freq = 50000
params.timesteps = 275000
params.gamma = 0.99
params.learning_rate = 0.0003
params.policy_kwargs = dict(activation_fn=th.nn.Tanh)
params.eps = 1
params.dt = 0.01
params.total_time = 5
params.trials = 3

#Env Specific Params

params.envs.pendulum.env = Pendulum.Pendulum() #base env for simulation
params.envs.pendulum.eval_env = Pendulum.Pendulum() #extra env for eval callback
params.envs.pendulum.run = False #if you want run_learning to train on this env
params.envs.pendulum.m = 1 #mass of pendulum
params.envs.pendulum.l = 1 #half the length of pendulum (length to com)
params.envs.pendulum.g = 5 #gravity
params.envs.pendulum.lam = 0.1 #damping coefficient
params.envs.pendulum.max_input = 5
params.envs.pendulum.min_input = -5
params.envs.pendulum.init_low = [-np.pi, -0.1]
params.envs.pendulum.init_high = [np.pi, 0.1]

params.envs.basependulum.env = BasePendulum.PendulumEnv #state = [th, th_dot], but obs = [sin(th), cos(th), th_dot]
params.envs.basependulum.eval_env = BasePendulum.PendulumEnv #extra env for eval callback
params.envs.basependulum.run = True

params.envs.baseenv.env = BaseEnv.BaseEnv #state = [th, th_dot], but obs = [sin(th), cos(th), th_dot]
params.envs.baseenv.eval_env = BaseEnv.BaseEnv #extra env for eval callback
params.envs.baseenv.max_input = 2
params.envs.baseenv.min_input = -2
params.envs.baseenv.init_low = -1
params.envs.baseenv.init_high = 1
params.envs.baseenv.run = False

params.envs.newpendulum.env = NewPendulum.Pendulum #state = [th, th_dot], but obs = [sin(th), cos(th), th_dot]
params.envs.newpendulum.eval_env = NewPendulum.Pendulum #extra env for eval callback
params.envs.newpendulum.run = False #if you want run_learning to train on this env
params.envs.newpendulum.m = 1 #mass of pendulum
params.envs.newpendulum.l = 1 #half the length of pendulum (length to com)
params.envs.newpendulum.g = 3 #gravity
params.envs.newpendulum.lam = 0.05 #damping coefficient
params.envs.newpendulum.max_input = 4
params.envs.newpendulum.min_input = -4
params.envs.newpendulum.init_low = [-np.pi, -0.3] #state, not obs
params.envs.newpendulum.init_high = [np.pi, 0.3]

params.envs.quadrotor.env = Quadrotor.Quadrotor()
params.envs.quadrotor.eval_env = Quadrotor.Quadrotor()
params.envs.quadrotor.run = False
params.envs.quadrotor.dt = params.dt
params.envs.quadrotor.total_time = params.total_time
params.envs.quadrotor.i_xx = 1
params.envs.quadrotor.m = 1
params.envs.quadrotor.g = 1
params.envs.quadrotor.max_input = np.array([10, 3])
params.envs.quadrotor.min_input = np.array([5, -3])
params.envs.quadrotor.init_low = -1
params.envs.quadrotor.init_high = 1

params.envs.pvtol.env = Pvtol.Pvtol()
params.envs.pvtol.eval_env = Pvtol.Pvtol()
params.envs.pvtol.run = False
params.envs.pvtol.dt = params.dt
params.envs.pvtol.total_time = params.total_time
params.envs.pvtol.eps = 0.01
params.envs.pvtol.m = 1
params.envs.pvtol.g = 2
params.envs.pvtol.max_input = np.array([4, 1])
params.envs.pvtol.min_input = np.array([1, -1])
params.envs.pvtol.init_low = -1
params.envs.pvtol.init_high = 1

params.envs.manipulator.env = Manipulator.Manipulator()
params.envs.manipulator.eval_env = Manipulator.Manipulator()
params.envs.manipulator.run = False
params.envs.manipulator.dt = params.dt
params.envs.manipulator.total_time = params.total_time
params.envs.manipulator.eps = 0.01
params.envs.manipulator.k1 = 1
params.envs.manipulator.k2 = 1
params.envs.manipulator.k3 = 1
params.envs.manipulator.max_input = 4
params.envs.manipulator.min_input = -4
params.envs.manipulator.init_low = -1
params.envs.manipulator.init_high = 1

params.envs.cartpole.env = Cartpole.Cartpole()
params.envs.cartpole.eval_env = Cartpole.Cartpole()
params.envs.cartpole.run = False
params.envs.cartpole.dt = params.dt
params.envs.cartpole.total_time = params.total_time
params.envs.cartpole.eps = 0.01
params.envs.cartpole.g = 1
params.envs.cartpole.mp = 0.5
params.envs.cartpole.mc = 1
params.envs.cartpole.l = 0.5  # actually half the pole's length
params.envs.cartpole.f = 5 # force
params.envs.cartpole.lam = 0.01
params.envs.cartpole.max_input = 2
params.envs.cartpole.min_input = -2
params.envs.cartpole.init_low = -1
params.envs.cartpole.init_high = 1
