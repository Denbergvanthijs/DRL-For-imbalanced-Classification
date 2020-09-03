import argparse
import os

import numpy as np
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

from data_pre import get_imb_data, load_data
from get_model import get_image_model, get_structured_model, get_text_model
from ICMDP_Env import ClassifyEnv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EPS_MAX = 1.0  # EpsGreedyQPolicy minimum
EPS_MIN = 0.1  # EpsGreedyQPolicy maximum
EPS_STEPS = 100_000  # Amount of steps to go (linear) from `EPS_MAX` to `EPS_MIN`
GAMMA = 0.5  # Discount factor
MODE = "train"  # Train or test mode
LR = 0.00025  # Learning rate
WARMUP_STEPS = 50_000  # Warmup period before training starts, https://stackoverflow.com/a/47455338
LOG_INTERVAL = 10_000  # Interval for logging, no effect on model performance
TARGET_MODEL_UPDATE = 10_000  # Frequency of updating the target network, https://github.com/keras-rl/keras-rl/issues/55

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

min_class = list(map(int, list(args.min_class)))
maj_class = list(map(int, list(args.maj_class)))

x_train, y_train, x_test, y_test = load_data(data_source)
x_train, y_train, x_test, y_test = get_imb_data(x_train, y_train, x_test, y_test, imb_rate, min_class, maj_class)
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"Minority: {min_class}, Majority: {maj_class}")

input_shape = x_train.shape[1:]
num_classes = len(set(y_test))
env = ClassifyEnv(MODE, imb_rate, x_train, y_train)

if args.model == "image":
    model = get_image_model(input_shape, num_classes)
elif args.model == "text":
    in_shape = [5_000, 500]
    model = get_text_model(in_shape, num_classes)
else:
    model = get_structured_model(input_shape, num_classes)

print(model.summary())


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
        processed_batch = batch.astype("float32") / 1.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


memory = SequentialMemory(limit=100_000, window_length=1)
processor = ClassifyProcessor()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps", value_max=EPS_MAX, value_min=EPS_MIN, value_test=.05, nb_steps=EPS_STEPS)
dqn = DQNAgent(model=model, policy=policy, nb_actions=num_classes, memory=memory, processor=processor,
               nb_steps_warmup=WARMUP_STEPS, gamma=GAMMA, target_model_update=TARGET_MODEL_UPDATE, train_interval=4, delta_clip=1.)

dqn.compile(Adam(lr=LR), metrics=["mae"])
dqn.fit(env, nb_steps=training_steps, log_interval=LOG_INTERVAL)

dqn.target_model.save("./models/mnistMin2MajAll.h5")

# Validation on train dataset
env.mode = "test"
dqn.test(env, nb_episodes=1, visualize=False)

# Validation on test dataset
env = ClassifyEnv(MODE, imb_rate, x_test, y_test)
env.mode = "test"
dqn.test(env, nb_episodes=1, visualize=False)
