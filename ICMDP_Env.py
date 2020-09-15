from datetime import datetime

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from tensorflow.compat.v1.summary import FileWriter, Summary

from utils import calculate_metrics, make_predictions


class ClassifyEnv(gym.Env):
    def __init__(self, mode, imb_rate, X_train, y_train, X_test, y_test, metrics_interval=10_000):
        """The custom classify environment."""
        self.mode = mode  # Train or Test mode
        self.imb_rate = imb_rate  # Imbalance rate: 0 < x < 1

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test  # Testdata used every `metrics_interval`-steps to calculate metrics
        self.y_test = y_test
        self.id = np.arange(self.X_train.shape[0])  # List of IDs to connect X and y data

        self.X_len = self.X_train.shape[0]

        self.action_space = spaces.Discrete(2)  # 2 classes: Minority and majority
        self.episode_step = 0  # Episode step, resets every episode

        self.writer = FileWriter("./logs/" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.step_number = 0  # Global episode number
        self.metrics_interval = metrics_interval  # Interval to update metrics for logging
        self.model = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        The environment takes a step based on action given by policy.
        Every `metrics_interval`-steps metrics will be logged for Tensorboard.
        """
        self.step_number += 1
        info = {}
        terminal = False
        curr_answer = self.y_train[self.id[self.episode_step]]

        if action == curr_answer:  # Correct action
            if curr_answer:  # Minority
                reward = 1
                # terminal = True
            else:  # Majority
                reward = self.imb_rate
        else:  # Incorrect action
            if curr_answer:  # Minority
                reward = -1
                if self.mode == "train":
                    terminal = True  # Stop episode when minority class is misclassified
            else:  # Majority
                reward = -self.imb_rate

        self.episode_step += 1

        if self.step_number % self.metrics_interval == 0:  # Collect metrics every `metrics_interval`-steps
            y_pred = make_predictions(self.model, self.X_test)
            info = calculate_metrics(self.y_test, y_pred)

            for k, v in info.items():
                summary = Summary(value=[Summary.Value(tag=k, simple_value=v)])
                self.writer.add_summary(summary, global_step=self.step_number)

        if self.episode_step == self.X_len - 1:
            terminal = True

        return self.X_train[self.id[self.episode_step]], reward, terminal, info

    def reset(self):
        """returns: (states, observations)."""
        if self.mode == "train":
            np.random.shuffle(self.id)

        self.episode_step = 0  # Reset episode step counter at the end of every episode
        return self.X_train[self.id[self.episode_step]]
