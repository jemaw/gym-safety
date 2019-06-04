"""A gridworld "game" whereby you try to navigate without hitting blocks

Openai Gym version
"""
# import curses
import gym
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import cv2



class GridNavigationEnv(gym.Env):
    def __init__(self):
        self.gridsize = None
        self.rho = None
        self.stochasticity = None
        self.threshold = None
        self.cumulat_constraint = 0
        self.state = None

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

        self.seed(42)
        self.make_game()
        self.fig = None

    def make_game(self, gridsize=6, rho=0.3, stochasticity=0.1, threshold=5):
        """Builds and returns a navigation game."""
        self.gridsize = gridsize
        self.rho = rho
        self.stochasticity = stochasticity
        self.threshold = threshold
        self.art = self.build_asci_art(gridsize, rho).reshape((gridsize, gridsize))

        # self.start_state = np.zeros_like(self.art, dtype=np.uint8)
        # self.start_state[-1, -1] = 1
        self.start_state = np.array([gridsize-1, gridsize -1])

        self.state = self.start_state

        self.art_img = self.get_art_img()

        self.ACTIONS = {
            0: np.array([-1,0]),
            1: np.array([0,-1]),
            2: np.array([1,0]),
            3: np.array([0,1])

        }

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

    def state_to_img(self):
        img = self.art_img.copy()
        i, j = self.state
        img[i,j,:] = self.FG_COLORS[self.PLAYER_CHAR]
        return img



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



    def render(self, mode='human'):
        """very minimalistic rendering"""
        img = self.state_to_img()
        img = cv2.resize(img, (self.display_size,self.display_size), interpolation=cv2.INTER_NEAREST)
        fig = plt.figure(0)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.00001)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



# class PlayerSprite(prefab_sprites.MazeWalker):
#     """
#     A `Sprite` for our player, the maze explorer.
#     This egocentric `Sprite` requires no logic beyond tying actions to
#     `MazeWalker` motion action helper methods, which keep the player from walking
#     on top of obstacles.
#     """
#     def __init__(self, corner, position, character):
#         """Constructor: player is egocentric and can't walk through walls."""
#         super(PlayerSprite, self).__init__(
#             corner, position, character, impassable='', confined_to_board=True)
#         # self._teleport(virtual_position)

#     def update(self, actions, board, layers, backdrop, things, the_plot):


#         # del backdrop, things, layers  # Unused
#         # fuel consumption
#         if actions != 5:
#             the_plot.add_reward(-1)
#             # stochasticity
#             if np.random.binomial(1, the_plot['stochasticity']):
#                 actions = np.random.randint(0,4)

#             if actions == 0:    # go upward?
#                 self._north(board, the_plot)
#             elif actions == 1:  # go downward?
#                 self._south(board, the_plot)
#             elif actions == 2:  # go leftward?
#                 self._west(board, the_plot)
#             elif actions == 3:  # go rightward?
#                 self._east(board, the_plot)
#             elif actions == 4:  # do nothing?
#                 the_plot.terminate_episode()
        

#             if layers[GOAL_CHAR][self.position]:
#                 the_plot.add_reward(1000)
#                 the_plot.terminate_episode()

#             if layers[OBSTACLE_CHAR][self.position]:
#                 the_plot['cumulat_constraint'] += 1
#                 if the_plot['cumulat_constraint'] >= the_plot['threshold']:
#                     the_plot.log('Threshold violated')
#                     # the_plot.add_reward(-1000)
#                     # the_plot.terminate_episode()


# def main(argv=()):
#   del argv  # Unused.


#   # Build a Hello World game.
#   game = make_game(gridsize, rho, stochasticity, threshold)

#   # Log a message in its Plot object.
#   game.the_plot.log('Hello, world!')

#   # Make a CursesUi to play it with.
#   ui = human_ui.CursesUi(
#       keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1, curses.KEY_LEFT: 2,
#                        curses.KEY_RIGHT: 3, 'q': 4, 'Q': 4, -1: 5},
#       delay=50, colour_fg=FG_COLORS)

#   # Let the game begin!
#   ui.play(game)
#   print(game.the_plot['cumulat_constraint'])


# if __name__ == '__main__':
#   main(sys.argv)
