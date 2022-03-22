"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class Cartpole(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Box(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, params, init=None):
        self.init = init
        self.global_params = params
        self.env_params = params.envs.cartpole
        self.total_mass = self.env_params.mp + self.env_params.mc
        self.polemass_length = self.env_params.mp * self.env_params.l
        self.kinematics_integrator = "euler"

        self.action_space = spaces.Box(low=self.env_params.min_input, high=self.env_params.max_input, shape=(1,), dtype=np.float32)
        
        high = np.array([self.env_params.thresh, 1, 1, 100000, 100000])
        self.observation_space = spaces.Box(low=-high, high=high,shape=(5,), dtype=np.float32)

        self.num_steps = self.global_params.total_time // self.global_params.dt
        self.viewer = None
        self.done = False

    def set_init(self, init):
        self.init = init

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):

        force = u[0]
        x, theta, x_dot, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.env_params.g * sintheta - costheta * temp) / (
            self.env_params.l * (4.0 / 3.0 - self.env_params.mp * costheta ** 2 / self.total_mass)
        ) - (self.env_params.lam * theta_dot / self.env_params.mp)
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.global_params.dt * x_dot
            x_dot = x_dot + self.global_params.dt * xacc
            theta = theta + self.global_params.dt * theta_dot
            theta_dot = theta_dot + self.global_params.dt * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.global_params.dt * xacc
            x = x + self.global_params.dt * x_dot
            theta_dot = theta_dot + self.global_params.dt * thetaacc
            theta = theta + self.global_params.dt * theta_dot

        self.state = (x, theta, x_dot, theta_dot)

        if (self.env_params.cost_func == 1):
            costs = self.get_cost1(u)
        elif (self.env_params.cost_func == 2):
            costs = self.get_cost2(u)

        self.curr_step += 1

        theta_normalized = self.angle_normalize(theta)

        self.done = bool(
            self.curr_step == self.num_steps)

        return self._get_obs, -costs, self.done, {}

    def _get_obs(self):
        x, theta, x_dot, thetadot = self.state
        return np.array([x, np.cos(theta), np.sin(theta), x_dot, thetadot])

    def get_cost1(self, u):

        theta = self.state[1]
        theta_dot = self.state[3]

        return (self.angle_normalize(theta)**2) + (self.global_params.eps*(u**2))

    def get_cost2(self, u):

        theta = self.state[1]
        theta_dot = self.state[3]

        return (self.angle_normalize(theta)**2) + (self.env_params.alpha*(self.global_params.eps**0.5)*(theta_dot**2)) + (self.global_params.eps*(u**2))

    def reset(self):
        if self.init:
            self.state = self.init
        else:
            self.state = np.random.uniform(self.env_params.init_low, self.env_params.init_high, (4,))

        self.curr_step = 0
        self.done = False
        return self._get_obs()

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.env_params.thresh * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.env_params.l)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[1])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def angle_normalize(self, x):
        return abs(((x + np.pi) % (2 * np.pi)) - np.pi)

