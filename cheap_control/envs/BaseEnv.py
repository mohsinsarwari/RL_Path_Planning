from typing import Optional
from os import path

import numpy as np


import gym
from gym import spaces
from gym.utils import seeding


class BaseEnv(gym.Env):
    """
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, params):
        self.params = params
        self.env_params = params.envs.baseenv

        self.num_steps = self.params.total_time // self.params.dt
        self.curr_step = 0

        high = 10000

        self.action_space = spaces.Box(low=self.env_params.min_input, high=self.env_params.max_input, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, shape=(1,), dtype=np.float32)

        self.state = np.random.uniform(self.env_params.init_low, self.env_params.init_high, (1,))

    def step(self, u):
        deriv = self.state[0] + u[0]
        self.state = [self.state[0] + (deriv * self.params.dt)]

        costs = self.state[0]**2 + self.params.eps*u[0]**2

        self.done = bool(
            self.curr_step == self.num_steps
            or abs(self.state[0]) > 1000000)

        return self._get_obs(), -costs, self.done, {}

    def _get_obs(self):
        return self.state

    def reset(self):
        self.state = np.random.uniform(self.env_params.init_low, self.env_params.init_high, (1,))
        self.last_u = None
        self.curr_step = 0
        return self._get_obs()