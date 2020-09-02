import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from sklearn.metrics import classification_report, confusion_matrix


class ClassifyEnv(gym.Env):
    def __init__(self, mode, imb_rate, trainx, trainy, ):
        """Mode means training or testing."""
        self.mode = mode
        self.imb_rate = imb_rate

        self.Env_data = trainx
        self.Answer = trainy
        self.id = np.arange(trainx.shape[0])

        self.game_len = self.Env_data.shape[0]

        self.num_classes = len(set(self.Answer))
        self.action_space = spaces.Discrete(self.num_classes)
        self.step_ind = 0
        self.y_pred = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        self.y_pred.append(a)
        y_true_cur = []
        info = {}
        terminal = False
        if a == self.Answer[self.id[self.step_ind]]:  # a: return of policy
            # When y_pred == y_true
            if self.Answer[self.id[self.step_ind]] == 0:  # Minority
                reward = 1.
            else:
                reward = 1. * self.imb_rate  # Majority
        else:
            # When y_pred != y_true
            if self.Answer[self.id[self.step_ind]] == 0:  # Minority
                reward = -1.
                if self.mode == 'train':
                    terminal = True  # Stop episode when minority class is misclassified
            else:
                reward = -1. * self.imb_rate  # Majority
        self.step_ind += 1

        if self.step_ind == self.game_len - 1:
            y_true_cur = self.Answer[self.id]
            info['gmean'], info['fmeasure'] = self.My_metrics(np.array(self.y_pred), np.array(y_true_cur[:self.step_ind]))
            terminal = True

        return self.Env_data[self.id[self.step_ind]], reward, terminal, info

    def My_metrics(self, y_pre, y_true):
        # Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        TN, FP, FN, TP = confusion_matrix(y_true, y_pre).ravel()

        precision = TP / (TP + FP)  # Positive Predictive Value
        recall = TP / (TP + FN)  # Sensitivity

        G_mean = np.sqrt(precision * recall)
        F_measure = 2 * (precision * recall / (precision + recall))

        print(classification_report(y_true, y_pre))
        print(f"TP: {TP} TN: {TN}\nFP: {FP} FN: {FN}")
        print(f"G-mean:{G_mean:.6f}, F_measure:{F_measure:.6f}\n")

        return G_mean, F_measure

    def reset(self):
        """returns: (states, observations)."""
        if self.mode == 'train':
            np.random.shuffle(self.id)

        self.step_ind = 0
        self.y_pred = []
        return self.Env_data[self.id[self.step_ind]]
