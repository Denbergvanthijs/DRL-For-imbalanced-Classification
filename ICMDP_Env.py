import gym
import numpy as np
from gym.utils import seeding


class ClassifyEnv(gym.Env):
    def __init__(self, mode, imb_rate, X_train, y_train):
        """The custom classify environment."""
        self.mode = mode  # Train or Test mode
        self.imb_rate = imb_rate  # Imbalance rate: 0 < x < 1

        self.X_train = X_train
        self.y_train = y_train

        self.X_len = self.X_train.shape[0]
        self.id = np.arange(self.X_len)  # List of IDs to connect X and y data

        self.episode_step = 0  # Episode step, resets every episode

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """The environment takes a step based on action given by policy."""
        curr_y_true = self.y_train[self.id[self.episode_step]]
        self.episode_step += 1
        terminal = False

        if action == curr_y_true:  # Correct action
            if curr_y_true:  # Minority
                reward = 1
            else:  # Majority
                reward = self.imb_rate

        else:  # Incorrect action
            if curr_y_true:  # Minority
                reward = -1
                if self.mode == "train":
                    terminal = True  # Stop episode when minority class is misclassified
            else:  # Majority
                reward = -self.imb_rate

        if self.episode_step == self.X_len - 1:
            terminal = True

        next_state = self.X_train[self.id[self.episode_step]]
        return next_state, reward, terminal, {}

    def reset(self):
        """returns: (states, observations)."""
        if self.mode == "train":
            np.random.shuffle(self.id)

        self.episode_step = 0  # Reset episode step counter at the end of every episode
        initial_state = self.X_train[self.id[self.episode_step]]
        return initial_state
