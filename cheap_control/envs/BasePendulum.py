from typing import Optional
from os import path

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class PendulumEnv(gym.Env):
    """
    ## Description
    The inverted pendulum swingup problem is a classic problem in the control literature. In this
    version of the problem, the pendulum starts in a random position, and the goal is to swing it up so
    it stays upright.
    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.
    ![Pendulum Coordinate System](./diagrams/pendulum.png)
    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta`: angle in radians.
    - `tau`: torque in `N * m`. Defined as positive _counter-clockwise_.
    ## Action Space
    The action is the torque applied to the pendulum.
    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |
    ## Observation Space
    The observations correspond to the x-y coordinate of the pendulum's end, and its angular velocity.
    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(angle)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |
    ## Rewards
    The reward is defined as:
    ```
    r = -(theta^2 + 0.1*theta_dt^2 + 0.001*torque^2)
    ```
    where `theta` is the pendulum's angle normalized between `[-pi, pi]`.
    Based on the above equation, the minimum reward that can be obtained is `-(pi^2 + 0.1*8^2 +
    0.001*2^2) = -16.2736044`, while the maximum reward is zero (pendulum is
    upright with zero velocity and no torque being applied).
    ## Starting State
    The starting state is a random angle in `[-pi, pi]` and a random angular velocity in `[-1,1]`.
    ## Episode Termination
    An episode terminates after 200 steps. There's no other criteria for termination.
    ## Arguments
    - `g`: acceleration of gravity measured in `(m/s^2)` used to calculate the pendulum dynamics. The default is
    `g=10.0`.
    ```
    gym.make('Pendulum-v1', g=9.81)
    ```
    ## Version History
    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, params, g=10.0, init=None):
        self.init = init
        self.params = params
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = self.params.dt
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.screen = None
        self.isopen = True
        self.viewer = None

        self.num_steps = self.params.total_time // self.params.dt
        self.curr_step = 0
        self.screen_dim = 500

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + self.params.eps * (u ** 2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])

        self.curr_step += 1

        self.done = bool(
            self.curr_step == self.num_steps)

        return self._get_obs(), -costs, self.done, {}

    def reset(self):
        high = np.array([np.pi, 1])
        if self.init:
            self.state = self.init
        else:
            self.state = np.random.uniform(low=-high, high=high)
        self.curr_step = 0
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-3, 3, -3, 3)

            length = self.l * 2
            width = self.l / 5

            l, r, t, b = -width / 2, width / 2, length, 0
            box = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
            circ0 = rendering.make_circle(width / 2)
            circ1 = rendering.make_circle(width / 2)
            circ1.add_attr(rendering.Transform(translation=(0, length)))
            rod = rendering.Compound([box, circ0, circ1])
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)

        self.pole_transform.set_rotation(self.state[0])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi