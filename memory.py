from __future__ import absolute_import

import itertools
import random
import warnings
from collections import deque, namedtuple

import numpy as np
from rl.memory import Memory


class PriorityMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(PriorityMemory, self).__init__(**kwargs)

        self.limit = limit
        self.minority_experiences = deque(maxlen=limit//2)
        self.majority_experiences = deque(maxlen=limit//2)
        self.last_class = None

    def sample(self, batch_size: int, batch_idxs=None):
        """Samples 50/50 minority and majority samples of size `batch_size`."""
        # Based on: https://stackoverflow.com/a/7064775/10603874
        # Slice all experiences except for last, since last experience is missing the next state
        if self.nb_entries <= batch_size:
            raise ValueError("Sample larger than memory")  # Since last state has no next state, batch_size can't be equal to memory size

        batch_min = batch_size // 2 + batch_size % 2
        batch_maj = batch_size // 2
        print(batch_min, batch_maj)

        if len(self.minority_experiences) < batch_min:
            batch_maj += batch_min - len(self.minority_experiences)
        if len(self.majority_experiences) < batch_maj:
            batch_min += batch_maj - len(self.majority_experiences)

        print(batch_min, batch_maj)
        experiences = []
        if self.last_class:  # Determine if last experience is minority or majority
            experiences += random.sample(self.majority_experiences, batch_maj)

            # Do not sample last experience since it is missing next_state
            min_slice = tuple(itertools.islice(self.minority_experiences, 0, len(self.minority_experiences) - 1))
            experiences += random.sample(min_slice, batch_min)
        else:
            experiences += random.sample(self.minority_experiences, batch_min)

            # Do not sample last experience since it is missing next_state
            max_slice = tuple(itertools.islice(self.majority_experiences, 0, len(self.majority_experiences) - 1))
            experiences += random.sample(max_slice, batch_maj)

        random.shuffle(experiences)
        return experiences

    def append(self, observation, action, reward, terminal, training=True):
        """Append experience to memory. Update the next state of the last appended experience."""
        super(PriorityMemory, self).append(observation, action, reward, terminal, training=training)

        current_class = True if reward in (-1, 1) else False

        if self.last_class is not None:  # If there's ever been an experience
            if self.last_class:  # If the last experience was a minority, set s' to the current state
                exp = self.minority_experiences.pop()
                exp[3] = observation  # s' is set to the current observation
                self.minority_experiences.append(exp)
            else:
                exp = self.majority_experiences.pop()
                exp[3] = observation  # s' is set to the current observation
                self.majority_experiences.append(exp)

        if current_class:  # Add current experience in the deque, check if experience is minority or majority
            self.minority_experiences.append([observation, action, reward, None, terminal])  # None is placeholder for s'
        else:
            self.majority_experiences.append([observation, action, reward, None, terminal])

        self.last_class = current_class  # True if current exp is minority, False if not

    @property
    def nb_entries(self):
        """Return number of observations."""
        return len(self.minority_experiences) + len(self.majority_experiences)

    @property
    def usefull_length(self):
        """Returns either length individual memories, depending if the last experience is """

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(PriorityMemory, self).get_config()
        config['limit'] = self.limit
        return config


if __name__ == "__main__":
    memory = PriorityMemory(limit=1_000, window_length=1)
    memory.append([1], 1, 1, True)  # s, a, r, t
    # print(memory.minority_experiences)
    # print(memory.majority_experiences)
    memory.append([1], 1, 1, False)
    memory.append([1], 1, 1, False)
    print(memory.minority_experiences)
    print(memory.majority_experiences)
    memory.append([0], 1, 0.5, False)
    memory.append([0], 1, 0.5, False)
    print(memory.minority_experiences, memory.majority_experiences)
    print(memory.sample(4))
