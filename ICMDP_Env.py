from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from gym import spaces
from gym.utils import seeding
from pandas import unique
from sklearn.metrics import confusion_matrix, mean_absolute_error


class ClassifyEnv(gym.Env):
    def __init__(self, mode, imb_rate, X_train, y_train):
        """The custom classify environment."""
        self.mode = mode  # Train or Test mode
        self.imb_rate = imb_rate  # Imbalance rate: 0 < x < 1

        self.X_train = X_train
        self.y_train = y_train
        self.id = np.arange(self.X_train.shape[0])  # List of IDs to connect X and y data

        self.game_len = self.X_train.shape[0]

        self.num_classes = unique(self.y_train).size
        self.action_space = spaces.Discrete(self.num_classes)
        self.step_ind = 0  # Episode step, resets every episode
        self.y_pred = []

        self.writer = tf.summary.FileWriter("./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.step_number = 0  # Global episode number

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        self.y_pred.append(a)  # Append by policy predicted answer to y_pred
        self.step_number += 1
        y_true_cur = []
        info = {}
        terminal = False
        curr_answer = self.y_train[self.id[self.step_ind]]

        if a == curr_answer:  # a: return of policy
            # When y_pred == y_true
            if curr_answer == 0:  # Minority
                reward = 1
            else:
                reward = self.imb_rate  # Majority
        else:
            # When y_pred != y_true
            if curr_answer == 0:  # Minority
                reward = -1
                if self.mode == "train":
                    terminal = True  # Stop episode when minority class is misclassified
            else:
                reward = -self.imb_rate  # Majority
        self.step_ind += 1

        if self.step_ind == self.game_len - 1:
            terminal = True

        if terminal is True:  # Collect metrics at the end of every episode.
            y_true_cur = self.y_train[self.id]
            info = self.metrics(np.array(y_true_cur[:self.step_ind]), np.array(self.y_pred))

            for k, v in info.items():
                summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v)])
                self.writer.add_summary(summary, global_step=self.step_number)

        return self.X_train[self.id[self.step_ind]], reward, terminal, info

    @staticmethod
    def metrics(y_true, y_pred):
        # Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        TP, FN, FP, TN = confusion_matrix(y_true, y_pred).ravel()  # Minority: positive, Majority: negative

        # Source: https://en.wikipedia.org/wiki/Precision_and_recall
        precision = TP / (TP + FP)  # Positive Predictive Value
        recall = TP / (TP + FN)  # Sensitivity, True Positive Rate (TPR)
        specificity = TN / (TN + FP)  # Specificity, selectivity, True Negative Rate (TNR)

        G_mean = np.sqrt(recall * specificity)  # Geometric mean of recall and specificity, defined in paper
        F_measure = np.sqrt(recall * precision)  # F-measure of recall and precision, defined in paper
        MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))  # Matthews Corr. Coefficient
        MAE = mean_absolute_error(y_true, y_pred)  # Mean absolute error

        return {"gmean": G_mean, "fmeasure": F_measure, "MCC": MCC, "MAE": MAE, "precision": precision, "recall": recall}

    def reset(self):
        """returns: (states, observations)."""
        if self.mode == "train":
            np.random.shuffle(self.id)

        self.step_ind = 0
        self.y_pred = []
        return self.X_train[self.id[self.step_ind]]
