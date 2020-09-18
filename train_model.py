import argparse
import os

import numpy as np
from keras.optimizers import Adam
from pandas import unique
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tensorflow.keras.models import load_model

from get_data import load_data
from get_model import get_image_model, get_structured_model, get_text_model
from ICMDP_Env import ClassifyEnv
from utils import make_predictions, plot_conf_matrix

# TODO: Determine why CPU is faster than GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # -1: Defaults to CPU, 0: GPU

EPS_MAX = 1.0  # EpsGreedyQPolicy maximum
EPS_MIN = 0.1  # EpsGreedyQPolicy minimum
EPS_STEPS = 200_000  # Amount of steps to go (linear) from `EPS_MAX` to `EPS_MIN`
GAMMA = 0.95  # Discount factor, importance of future reward
LR = 0.00025  # Learning rate
WARMUP_STEPS = 60_000  # Warmup period before training starts, https://stackoverflow.com/a/47455338
TARGET_MODEL_UPDATE = 0.0005  # Frequency of updating the target network, https://github.com/keras-rl/keras-rl/issues/55
MEMORY_SIZE = 100_000  # Size of the SequentialMemory
BATCH_SIZE = 32  # Minibatch size sampled from SequentialMemory
DOUBLE_DQN = False  # To enable or disable DDQN as proposed by https://arxiv.org/pdf/1509.06461.pdf
NORMALIZATION = False  # Normalize the Kaggle Credit Card Fraud dataset?
MODE = "train"  # Train or test mode
LOG_INTERVAL = 60_000  # Interval for logging, no effect on model performance
FP_MODEL = "./models/credit.h5"  # Filepath to save the trained model

parser = argparse.ArgumentParser()
parser.add_argument("--data", choices=["mnist", "cifar10", "famnist", "imdb", "credit"], default="mnist")
parser.add_argument("--model", choices=["image", "text", "structured"], default="image")
parser.add_argument("--imb-rate", type=float, default=0.04)
parser.add_argument("--min-class", type=str, default="2")
parser.add_argument("--maj-class", type=str, default="3")
parser.add_argument("--training-steps", type=int, default=10_000)
args = parser.parse_args()

data_source = args.data
imb_rate = args.imb_rate
training_steps = args.training_steps

min_class = list(map(int, args.min_class))  # String to list of integers
maj_class = list(map(int, args.maj_class))  # String to list of integers

X_train, y_train, X_test, y_test, X_val, y_val = load_data(data_source, imb_rate, min_class, maj_class, normalization=NORMALIZATION)
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Minority: {min_class}, Majority: {maj_class}")

input_shape = X_train.shape[1:]
env = ClassifyEnv(MODE, imb_rate, X_train, y_train, X_test, y_test)

if args.model == "image":
    model = get_image_model(input_shape)
elif args.model == "text":
    input_shape = (5_000, 500)
    model = get_text_model(input_shape)
else:
    model = get_structured_model(input_shape)

# print(model.summary())


class ClassifyProcessor(Processor):
    def process_observation(self, observation):
        if args.model == "text":
            return observation

        img = observation.reshape(input_shape)
        processed_observation = np.array(img)
        return processed_observation

    def process_state_batch(self, batch):
        if args.model == "text":
            return batch.reshape((-1, input_shape[1]))

        batch = batch.reshape((-1,) + input_shape)
        return batch.astype("float32") / 1

    def process_reward(self, reward):
        return np.clip(reward, -1, 1)


processor = ClassifyProcessor()
memory = SequentialMemory(limit=MEMORY_SIZE, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps", value_max=EPS_MAX, value_min=EPS_MIN, value_test=0.05, nb_steps=EPS_STEPS)
dqn = DQNAgent(model=model, policy=policy, nb_actions=2, memory=memory, processor=processor, nb_steps_warmup=WARMUP_STEPS, gamma=GAMMA,
               target_model_update=TARGET_MODEL_UPDATE, train_interval=4, delta_clip=1, batch_size=BATCH_SIZE, enable_double_dqn=DOUBLE_DQN)

dqn.compile(Adam(lr=LR), metrics=["mae"])
env.model = model  # Set the prediction model for the environment. Used to calculate metrics
dqn.fit(env, nb_steps=training_steps, log_interval=LOG_INTERVAL)
dqn.target_model.save(FP_MODEL)

# Validate on validation dataset
trained_model = load_model(FP_MODEL)  # Load the just saved model
y_pred = make_predictions(trained_model, X_val)
plot_conf_matrix(y_val, y_pred)
