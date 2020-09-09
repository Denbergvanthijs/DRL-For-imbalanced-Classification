from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from gym import spaces
from gym.utils import seeding
from pandas import unique

from predict import calculate_metrics


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

        self.game_len = self.X_train.shape[0]

        self.num_classes = unique(self.y_train).size
        self.action_space = spaces.Discrete(self.num_classes)
        self.step_ind = 0  # Episode step, resets every episode
        self.y_pred = []

        self.writer = tf.compat.v1.summary.FileWriter("./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.step_number = 0  # Global episode number
        self.metrics_interval = metrics_interval  # Interval to update metrics for logging

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        self.y_pred.append(a)  # Append by policy predicted answer to y_pred
        self.step_number += 1
        info = {}
        terminal = False
        curr_answer = self.y_train[self.id[self.step_ind]]

        if a == curr_answer:  # a: return of policy
            # When y_pred == y_true
            if curr_answer == 1:  # Minority
                reward = 1
            else:
                reward = self.imb_rate  # Majority
        else:
            # When y_pred != y_true
            if curr_answer == 1:  # Minority
                reward = -1
                if self.mode == "train":
                    terminal = True  # Stop episode when minority class is misclassified
            else:
                reward = -self.imb_rate  # Majority
        self.step_ind += 1

        if self.step_ind == self.game_len - 1:
            terminal = True

        if self.step_number % self.metrics_interval == 0:  # Collect metrics every `metrics_interval`-steps
            y_pred = [np.argmax(x) for x in self.model.predict(self.X_test)]
            info = calculate_metrics(self.y_test, y_pred)

            for k, v in info.items():
                summary = tf.compat.v1.summary.Summary(value=[tf.compat.v1.summary.Summary.Value(tag=k, simple_value=v)])
                self.writer.add_summary(summary, global_step=self.step_number)

        return self.X_train[self.id[self.step_ind]], reward, terminal, info

    def reset(self):
        """returns: (states, observations)."""
        if self.mode == "train":
            np.random.shuffle(self.id)

        self.step_ind = 0
        self.y_pred = []
        return self.X_train[self.id[self.step_ind]]
