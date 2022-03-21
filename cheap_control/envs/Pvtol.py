import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class Pvtol(gym.Env):

    """
    Description:
        PVTOL model with 2 inputs: thrust and torque about its center. Gravity is
        constantly pulling the PVTOL down. (imagine plane going into the screen)

    Source:
        ???

    Observation:
        Type: Box(6)
        Num     Observation               Min                     Max
        0       x-position                -5                       5
        1       y-position                -5                       5
        2       cos(theta)               -1                        1
        3       sin(theta)               -1                        1
        4       x-dot                    -100000                  100000
        5       y-dot                    -100000                  100000
        6       theta-dot                -100000                  100000

    Actions:
        Type: Box(2)
        Num   Action
        0     Thrust
        1     Torque

    Reward:
        Cost function part of params

    Starting State:
        All observations are assigned a uniform random value in [-1..1]

    Episode Termination:
        ????
        Time based
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self,  params, init=None):
        self.curr_eval = 0
        self.init = init
        self.global_params = params
        self.env_params = params.envs.pvtol
        self.action_space = spaces.Box(low=self.env_params.min_input, high=self.env_params.max_input, shape=(2,), dtype=np.float32)
        
        high = np.array([self.env_params.thresh, self.env_params.thresh, 1, 1, 100000, 100000, 100000])
        self.observation_space = spaces.Box(low=-high, high=high,shape=(7,), dtype=np.float32)

        self.num_steps = self.global_params.total_time // self.global_params.dt
        self.viewer = None
        self.done = False

    def set_init(self, init):
        self.init = init

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
                                (-np.sin(theta)*u1) + (self.env_params.k*np.cos(theta)*u2),
                                (np.cos(theta)*u1) + (self.env_params.k*np.sin(theta)*u2) - self.env_params.g,
                                u2])

        self.state = self.state + (self.global_params.dt * derivatives)

        if (self.env_params.cost_func == 1):
            costs = self.get_cost1(u)

        self.curr_step += 1

        self.done = bool(
            self.curr_step == self.num_steps)

        return self._get_obs(), -costs, self.done, {}

    def get_cost1(self, u):

        u1 = u[0]
        u2 = u[1]
        x = self.state[0]
        y = self.state[1]

        return (x**2) + (y**2) + (self.global_params.eps*((u1-self.env_params.g)**2 + u2**2))

    def reset(self):
        if self.init:
            self.state = self.init
        else:
            self.state = np.random.uniform(self.env_params.init_low, self.env_params.init_high, (6,))
        self.curr_step = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        x, y, theta, x_dot, y_dot, theta_dot = self.state
        return np.array([x, y, np.cos(theta), np.sin(theta), x_dot, y_dot, theta_dot])

    def render(self, mode="human"):
        screen_width = 400
        screen_height = 400

        world_width = self.env_params.thresh * 2
        scalex = screen_width // world_width
        scaley = screen_height // world_width
        centery = screen_height // 2  # TOP OF pvtol
        centerx = screen_width // 2 
        pvtolwidth = 100.0
        pvtolheight = 25.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -pvtolwidth / 2, pvtolwidth / 2, pvtolheight / 2, -pvtolheight / 2
            pvtol = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.pvtoltrans = rendering.Transform()
            pvtol.add_attr(self.pvtoltrans)
            self.viewer.add_geom(pvtol)
            self.trackx = rendering.Line((0, centery), (screen_width, centery))
            self.trackx.set_color(0, 0, 0)
            self.viewer.add_geom(self.trackx)
            self.tracky = rendering.Line((centerx, 0), (centerx, screen_height))
            self.tracky.set_color(0, 0, 0)
            self.viewer.add_geom(self.tracky)

        if self.state is None:
            return None

        x = self.state
        pvtolx = x[0] * scalex + centerx # MIDDLE OF pvtol
        pvtoly = x[1] * scaley + centery
        theta = x[2]
        self.pvtoltrans.set_translation(pvtolx, pvtoly)
        self.pvtoltrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def angle_normalize(self, x):
            return abs(((x + np.pi) % (2 * np.pi)) - np.pi)