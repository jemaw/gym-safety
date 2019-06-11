"""A gridworld "game" whereby you try to navigate without hitting blocks

Openai Gym version
"""
# import curses
import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import cv2

DEFAULT_SEED = 42

class GridNavigationEnv(gym.Env):
    """"
    Description:
        This game is used as simple environment for testing reinforcement learning algorithms that are built for safety.
        It is an unnoficial implementation of the one described in "A Lyapunov-based Approach to Safe Reinforcement Learning":
            https://arxiv.org/abs/1805.07708
        The goal for the player is to reach the end-state without violating constraints, here these constraints are modelled as
        immediate cost when hitting an obstacle and the constraint is to have the cumulative constraint cost below a certain threshold.
        The environment is stochastic, sometimes the agent performs a random action, instead of the chosen one.

    Parameters for customize game:
        seed (float):               seed for creating the game, can also be None for random seed
        rho (float):                density, meaning probability for obstacle, default: 0.3
        stochasticity (float):      probability for random action, default: 0.1
        img_observation (bool):     whether to use an img_observation or a one-hot encoding of the player position, default: false

        Note: v0 has gridsize 25 and v1 has grisize 60 for further customization use the customize_game function

    Observation:
        this environment has two modes: the observation is either a one-hot encoding of the player position or an RGB image of the complete board


    Actions:
        Type: Discrete(4)
        Num	Action
        0	Go Down
        1	Go Left
        2       Go up
        3       Go right

    Reward:
        Every action has a negative fuel reward of -1. If the final state is reached the reward is 1000.

    Constraint Cost:
        If an obstacle is hit the constraint cost is 1, else 0


    Starting State:
        The agent always starts at the bottom right

    Epsiode Termination:
        - the agent reaches the goal state
        - Episode length is greater than 200


    """
    def __init__(self, gridsize=32):
        self.gridsize = gridsize
        self.rho = 0.3
        self.stochasticity = 0.1
        self.img_observation = False
        self.cumulat_constraint = 0
        self.state = None
        self.start_state = None
        self.goal_state = None
        self.img_observation = None

        self.action_space = None
        self.observation_space = None


        self.EMPTY_CHAR = ' '
        self.OBSTACLE_CHAR = '#'
        self.PLAYER_CHAR = 'P'
        self.GOAL_CHAR = 'G'

        self.display_size = 256
        self.FG_COLORS = {self.EMPTY_CHAR: (0, 0, 0),  # normal background
                         self.GOAL_CHAR: (0, 1, 0),  # goal
                         self.PLAYER_CHAR: (1, 0, 0),   # player
                         self.OBSTACLE_CHAR: (0, 0, 1),      # obstacle
                     }

        self.ACTIONS = {
            0: np.array([-1,0]),
            1: np.array([0,-1]),
            2: np.array([1,0]),
            3: np.array([0,1])

        }
        self.seed(DEFAULT_SEED)
        self.make_game()

    def customize_game(self, seed=DEFAULT_SEED, rho=None, stochasticity=None, img_observation=None):
        self.seed(seed)
        if not rho is None:
            self.rho = rho
        if not stochasticity is None:
            self.stochasticity = stochasticity
        if not img_observation is None:
            self.img_observation = img_observation
        self.make_game()


    def make_game(self):
        """Builds and returns a navigation game."""
        self.art = self.build_asci_art(self.gridsize, self.rho).reshape((self.gridsize, self.gridsize))

        self.start_state = np.array([self.gridsize-1, self.gridsize -1])
        self.goal_state = np.argwhere(self.art == self.GOAL_CHAR)[0]

        self.state = self.start_state
        self.obstacle_states = np.argwhere(self.art == self.OBSTACLE_CHAR).tolist()

        self.art_img = self.get_art_img()

        self.action_space = spaces.Discrete(4)
        if self.img_observation:
            self.observation_space = spaces.Box(0, 1, shape=(self.gridsize, self.gridsize, 3), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(0, 1, shape=(self.gridsize*self.gridsize,), dtype=np.float32)

    def build_asci_art(self, gridsize, rho):
        art = []
        first_row = [' ']*gridsize
        alpha = self.np_random.randint(0, gridsize)
        first_row[alpha] = self.GOAL_CHAR
        first_row = ''.join(first_row)
        last_row = [' ']*gridsize
        last_row[-1] = self.PLAYER_CHAR
        last_row = ''.join(last_row)
        art.append(first_row)
        for row_idx in range(gridsize-2):
            row = [' ']*gridsize
            for col_idx in range(gridsize):
                if self.np_random.binomial(1, rho):
                    row[col_idx] = self.OBSTACLE_CHAR
            art.append(''.join(row))
        art.append(last_row)
        split_art = []
        for row in art:
            for ch in row:
                split_art.append(ch)
        return np.array(split_art)

    def get_art_img(self):
        """without player"""
        img = np.zeros((self.gridsize, self.gridsize, 3))
        for i in range(self.gridsize):
            for j in range(self.gridsize):
                if self.art[i,j] == self.PLAYER_CHAR:
                    img[i,j,:] = self.FG_COLORS[self.EMPTY_CHAR]
                else:
                    img[i,j,:] = self.FG_COLORS[self.art[i,j]]
        return img




    def reset(self):
        self.state = self.start_state
        if self.img_observation:
            obs = self.state_to_img()
        else:
            obs = self.state_to_oh()
        return obs

    def state_to_img(self):
        img = self.art_img.copy()
        i, j = self.state
        img[i,j,:] = self.FG_COLORS[self.PLAYER_CHAR]
        return img

    def state_to_oh(self):
        obs = np.zeros((self.gridsize*self.gridsize))
        state_i, state_j = self.state
        obs[state_i*self.gridsize + state_j] = 1
        return obs



    def step(self, action):
        if np.random.binomial(1, self.stochasticity):
            action = np.random.randint(0,4)

        new_state = self.state+self.ACTIONS[action]

        new_state_x, new_state_y = new_state
        if not (new_state_x < 0 or \
           new_state_x >= self.gridsize or \
           new_state_y < 0 or \
           new_state_y >= self.gridsize):
            self.state = new_state

        reward = -1
        done = False
        info = {'constraint_costs':[0]}

        if (self.state == self.goal_state).all():
            done = True
            reward = 1000
        elif self.state.tolist() in self.obstacle_states:
            info['constraint_costs'] = [1]


        if self.img_observation:
            obs = self.state_to_img()
        else:
            obs = self.state_to_oh()

        return obs, reward, done, info


    def render(self, mode='human'):
        """very minimalistic rendering"""
        img = self.state_to_img()
        img = cv2.resize(img, (self.display_size,self.display_size), interpolation=cv2.INTER_NEAREST)
        if mode == 'human':
            fig = plt.figure(0)
            plt.clf()
            plt.imshow(img)
            fig.canvas.draw()
            plt.pause(0.00001)
        return img


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
