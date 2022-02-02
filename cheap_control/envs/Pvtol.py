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
        0       x-position                ??                       ??
        1       y-position               ??                        ??
        2       theta                    -2pi                     2pi
        3       x-dot                    ??                       ??
        4       y-dot                    ??                       ??
        5       theta-dot                ??                       ??

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

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params
        self.action_space = spaces.Box(low=params.min_input, high=params.max_input, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=params.min_state, high=params.max_state,shape=(6,), dtype=np.float32)
        self.num_steps = self.params.total_time // self.params.dt
        params.seed = self.seed()
        self.viewer = None
        self.done = False

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
                                (-np.sin(theta)*u1) + (self.params.eps*np.cos(theta)*u2),
                                (np.cos(theta)*u1) + (self.params.eps*np.sin(theta)*u2) - self.params.g,
                                u2])

        self.state = self.state + (self.params.dt * derivatives)
        self.state[2] = self.state[2] % (2*np.pi)

        costs = self.get_cost(u)

        self.curr_step += 1

        theta_normalized = self.angle_normalize(theta)

        self.done = bool(
            self.curr_step == self.num_steps
            or theta_normalized > (np.pi / 2))

        return self.state, -costs, self.done, {}

    def get_cost(self, u):

        theta = self.angle_normalize(self.state[2])
        theta_dot = self.state[5]

        return (theta**2) + (self.env_params.eps*(u**2))

    def reset(self):
        self.state = np.random.uniform(self.params.init_low, self.params.init_high, (6,))
        self.curr_step = 0
        self.done = False
        return self.state

    def render(self, mode="human"):
        screen_width = 400
        screen_height = 400

        world_width = self.params.max_state * 2
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