import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class QuadrotorEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, param_dict, i_xx=1, m=1, g=1):
        self.dt = param_dict["dt"]
        self.num_steps = param_dict["total_time"] // self.dt
        self.curr_step = 0
        self.i_xx = i_xx
        self.m = m
        self.g = g
        self.max_input = 5
        self.min_input = 0
        self.max_state = 5
        self.min_state = -5
        self.viewer = None
        self.done = False

        self.action_space = spaces.Box(low=self.min_input, high=self.max_input, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_state, high=self.max_state,shape=(6,), dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        u1 = u[0]
        u2 = u[1]
        x = self.state[0]
        y = self.state[1]
        theta = self.state[2]
        x_dot = self.state[3]
        y_dot = self.state[4]
        theta_dot = self.state[5]

        derivatives = np.array([x_dot, y_dot, theta_dot, 
                                (-(1/self.m)*np.sin(theta)*u1),
                                ((1/self.m)*np.cos(theta)*u1) - self.g,
                                (1/self.i_xx)*u2])

        self.state = self.state + (self.dt * derivatives)

        costs = (self.state[1] - 2)**2 + (self.state[0] - 2)**2

        self.curr_step += 1
        if self.curr_step == self.num_steps:
            self.done = True
 
        return self.state, -costs, self.done, {}

    def reset(self):
        self.state = np.zeros(6)
        self.curr_step = 0
        self.done = False
        return self.state

    def render(self, mode="human"):
        screen_width = 800
        screen_height = 800

        world_width = self.max_input * 2
        scalex = screen_width / world_width
        scaley = screen_height / world_width
        centery = screen_height / 2  # TOP OF CART
        centerx = screen_width / 2 
        cartwidth = 100.0
        cartheight = 25.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            self.trackx = rendering.Line((0, centery), (screen_width, centery))
            self.trackx.set_color(0, 0, 0)
            self.viewer.add_geom(self.trackx)
            self.tracky = rendering.Line((centerx, 0), (centerx, screen_height))
            self.tracky.set_color(0, 0, 0)
            self.viewer.add_geom(self.tracky)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scalex + screen_width / 2.0  # MIDDLE OF CART
        carty = x[1] * scaley + screen_height / 2.0
        theta = x[2]
        self.carttrans.set_translation(cartx, carty)
        self.carttrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None