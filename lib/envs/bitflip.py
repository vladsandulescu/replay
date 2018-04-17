import numpy as np
import gym

class BitFlipEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    np.random.seed(42)

    def __init__(self, n):
        self.n = n
        self.state = None
        self.done = False

    def step(self, action, goal):
        if self.state is None:
            raise Exception('You must first initialize the environment by calling the reset method.')

        self.state[action] = 1 if self.state[action] == 0 else 0
        self.done = np.array_equal(self.state, goal)
        reward = -1 if not self.done else 0
        return self.state.copy(), reward, self.done, {}

    def reset(self):
        self.state = np.random.choice([0, 1], size=self.n)
        return self.state.copy()

    def render(self, mode='human', close=False):
        pass
