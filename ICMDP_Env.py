from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from gym import spaces
from gym.utils import seeding
from pandas import unique
from sklearn.metrics import classification_report, confusion_matrix


class ClassifyEnv(gym.Env):
    def __init__(self, mode, imb_rate, trainx, trainy):
        """Mode means training or testing."""
        self.mode = mode
        self.imb_rate = imb_rate

        self.Env_data = trainx
        self.Answer = trainy
        self.id = np.arange(trainx.shape[0])

        self.game_len = self.Env_data.shape[0]

        self.num_classes = unique(self.Answer).size
        self.action_space = spaces.Discrete(self.num_classes)
        self.step_ind = 0
        self.y_pred = []

        self.writer = tf.summary.FileWriter("./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.episode = 0  # The episode number

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        self.y_pred.append(a)
        y_true_cur = []
        info = {}
        terminal = False
        curr_answer = self.Answer[self.id[self.step_ind]]

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
            self.My_metrics(np.array(self.y_pred), np.array(y_true_cur[:self.step_ind]), final=True)  # Print final metrics
            terminal = True

        if terminal is True:  # Collect metrics at the end of every episode.
            self.episode += 1
            y_true_cur = self.Answer[self.id]
            info = self.My_metrics(np.array(self.y_pred), np.array(y_true_cur[:self.step_ind]))

            for metric in zip(("gmean", "fmeasure", "MCC"), (info["gmean"], info["fmeasure"], info["MCC"])):
                summary = tf.Summary(value=[tf.Summary.Value(tag=metric[0], simple_value=metric[1])])
                self.writer.add_summary(summary, global_step=self.episode)

        return self.Env_data[self.id[self.step_ind]], reward, terminal, info

    def My_metrics(self, y_pred, y_true, final=False):
        # Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        TP, FN, FP, TN = confusion_matrix(y_true, y_pred).ravel()  # Minority: positive, Majority: negative

        # Source: https://en.wikipedia.org/wiki/Precision_and_recall
        precision = TP / (TP + FP)  # Positive Predictive Value
        recall = TP / (TP + FN)  # Sensitivity, True Positive Rate (TPR)
        specificity = TN / (TN + FP)  # Specificity, selectivity, True Negative Rate (TNR)

        G_mean = np.sqrt(recall * specificity)  # Geometric mean of recall and specificity, defined in paper
        F_measure = np.sqrt(recall * precision)  # F-measure of recall and precision, defined in paper
        MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))  # Matthews Corr. Coefficient

        if final:  # Only print metrics when training is complete
            print(classification_report(y_true, y_pred, target_names=["Minority", "Majority"]))
            print(f"TP: {TP} TN: {TN}\nFP: {FP} FN: {FN}")
            print(f"G-mean:{G_mean:.6f}, F_measure:{F_measure:.6f}, MCC: {MCC:.6f}\n")

        return {"gmean": G_mean, "fmeasure": F_measure, "MCC": MCC}

    def reset(self):
        """returns: (states, observations)."""
        if self.mode == "train":
            np.random.shuffle(self.id)

        self.step_ind = 0
        self.y_pred = []
        return self.Env_data[self.id[self.step_ind]]
