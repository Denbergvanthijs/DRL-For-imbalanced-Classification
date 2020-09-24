import csv
import os
from datetime import datetime

import numpy as np
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

from callbacks import Metrics
from get_data import load_data
from get_model import get_structured_model
from ICMDP_Env import ClassifyEnv
from utils import calculate_metrics, make_predictions

# TODO: Determine why CPU is faster than GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # -1: Defaults to CPU, 0: GPU
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"  # Support for Accelerated Linear Algebra (XLA)

EPS_MAX = 1.0  # EpsGreedyQPolicy maximum
EPS_MIN = 0.05  # EpsGreedyQPolicy minimum
EPS_STEPS = 200_000  # Amount of steps to go (linear) from `EPS_MAX` to `EPS_MIN`
GAMMA = 0.95  # Discount factor, importance of future reward
LR = 0.001  # Learning rate
WARMUP_STEPS = 60_000  # Warmup period before training starts, https://stackoverflow.com/a/47455338
TARGET_MODEL_UPDATE = 0.0005  # Frequency of updating the target network, https://github.com/keras-rl/keras-rl/issues/55
MEMORY_SIZE = 100_000  # Size of the SequentialMemory
BATCH_SIZE = 32  # Minibatch size sampled from SequentialMemory
DOUBLE_DQN = True  # To enable or disable DDQN as proposed by https://arxiv.org/pdf/1509.06461.pdf
NORMALIZATION = False  # Normalize the Kaggle Credit Card Fraud dataset?
MODE = "train"  # Train or test mode
LOG_INTERVAL = 60_000  # Interval for logging, no effect on model performance

TRAINING_STEPS = 500_000
min_class = [1]
maj_class = [0]
imb_rate = 0.01
input_shape = (29,)
columns = ["Gmean", "Fdot5", "F1", "F2", "TP", "TN", "FP", "FN"]
N_REPETITIONS = 100
LOG_EVERY = 10
model = get_structured_model(input_shape)


class ClassifyProcessor(Processor):
    def process_observation(self, observation):
        img = observation.reshape(input_shape)
        processed_observation = np.array(img)
        return processed_observation

    def process_state_batch(self, batch):
        batch = batch.reshape((-1,) + input_shape)
        return batch.astype("float32") / 1

    def process_reward(self, reward):
        return np.clip(reward, -1, 1)


processor = ClassifyProcessor()
with open(f"./logs_alt/DQN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for i in range(N_REPETITIONS):
        X_train, y_train, X_test, y_test, X_val, y_val = load_data("credit", imb_rate, min_class, maj_class, normalization=NORMALIZATION)
        env = ClassifyEnv(MODE, imb_rate, X_train, y_train)
        memory = SequentialMemory(limit=MEMORY_SIZE, window_length=1)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps", value_max=EPS_MAX,
                                      value_min=EPS_MIN, value_test=0.05, nb_steps=EPS_STEPS)

        dqn = DQNAgent(model=model, policy=policy, nb_actions=2, memory=memory, processor=processor, nb_steps_warmup=WARMUP_STEPS, gamma=GAMMA,
                       target_model_update=TARGET_MODEL_UPDATE, train_interval=4, delta_clip=1, batch_size=BATCH_SIZE, enable_double_dqn=DOUBLE_DQN)
        dqn.compile(Adam(lr=LR))

        metrics = Metrics(X_val, y_val)
        dqn.fit(env, nb_steps=TRAINING_STEPS, log_interval=LOG_INTERVAL, callbacks=[metrics], verbose=0)
        y_pred = make_predictions(dqn.target_model, X_test)
        stats = calculate_metrics(y_test, y_pred)  # Get stats as dictionairy
        writer.writerow(stats)  # Write dictionairy as row

        if not i % LOG_EVERY:
            print(f"{i}: FN: {stats.get('FN')}, FP: {stats.get('FP')}")
