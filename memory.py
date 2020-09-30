import itertools
import math
import random

import numpy as np
from rl.memory import Experience, Memory


class PriorityMemory(Memory):
    """
    Saves and samples minority and majority samples.
    Memory is based on two rotating lists.
    Partly based on keras-rl's SequentialMemory.
    Partly based of: https://stackoverflow.com/a/56171119/10603874
    """

    def __init__(self, limit, minority_chance: float = 0.5, **kwargs):
        super(PriorityMemory, self).__init__(**kwargs)

        self.limit = limit
        self.minority_chance = minority_chance

        if isinstance(self.limit, int):
            self.min_limit = self.limit // 2
            self.maj_limit = self.limit // 2
        elif isinstance(self.limit, (tuple, list)):
            self.min_limit = self.limit[0]
            self.maj_limit = self.limit[1]
        else:
            # TODO: Add checking for tuple length.
            raise TypeError("`limit` must be a tuple of length 2 or an integer.")

        self.min_experiences = np.full(self.min_limit, fill_value=None)
        self.maj_experiences = np.full(self.maj_limit, fill_value=None)
        self.min_index = 0
        self.min_size = 0
        self.maj_index = 0
        self.maj_size = 0
        self.last_class = None

    def sample(self, batch_size: int, batch_idxs=None):
        """
        Samples minority and majority samples of size `batch_size`.
        Ratio between minority/majority is determined by `minority_chance`.

        If there's not enough data in one of the batches, extra data from other batch will be used.
        """
        if self.nb_entries <= batch_size:
            # Since last state has no next state, `batch_size` can't be equal to memory size
            raise ValueError("Sample larger or equal to memory")

        # Initial batch sizes, checking will be performed to ensure there's enough data in each batch
        batch_min, batch_maj = self.calculate_batch_sizes(batch_size, self.minority_chance)

        if self.last_class:  # Remove last added experience since it has no s'
            last_index = (self.min_index - 1) % self.min_limit
            min_slice = self.min_experiences[np.arange(self.min_experiences.size) != last_index]
            min_slice = min_slice[min_slice != None]

            maj_slice = self.maj_experiences[self.maj_experiences != None]
        else:
            last_index = (self.maj_index - 1) % self.maj_limit
            maj_slice = self.maj_experiences[np.arange(self.maj_experiences.size) != last_index]
            maj_slice = maj_slice[maj_slice != None]

            min_slice = self.min_experiences[self.min_experiences != None]

        len_min, len_maj = min_slice.size, maj_slice.size

        if len_min < batch_min:  # If not enough data in memory of specific class
            batch_maj += batch_min - len_min  # Get extra data from other class
            batch_min = len_min
        if len_maj < batch_maj:
            batch_min += batch_maj - len_maj
            batch_maj = len_maj

        assert batch_size == batch_min + batch_maj, "This should not happen"

        # If any batch was not the same as the corresponding useful length, slices will be the same as original list's
        minibatch = np.concatenate([np.random.choice(min_slice, batch_min), np.random.choice(maj_slice, batch_maj)])
        np.random.shuffle(minibatch)
        return minibatch

    def append(self, observation, action, reward, terminal, training=True):
        """
        Append experience to memory.
        Updates s' of the last appended experience.
        """
        super(PriorityMemory, self).append(observation, action, reward, terminal, training=training)

        if not training:  # If testing
            return

        current_class = True if reward in (-1, 1) else False  # True if current experience is minority, False if not

        if self.last_class is None:  # Initial value of `last_class` is None
            pass
        elif self.last_class:  # If the last experience was a minority, set s' to the current state
            last_index = (self.min_index - 1) % self.min_limit
            s, a, r, _, t = self.min_experiences[last_index]  # Unpacking since tuples are inmutable
            exp = Experience(state0=s, action=a, reward=r, state1=observation, terminal1=t)  # s' is set to the current observation
            self.min_experiences[last_index] = exp  # Overwrite last experience
        elif not self.last_class:
            last_index = (self.maj_index - 1) % self.maj_limit
            s, a, r, _, t = self.maj_experiences[last_index]  # Unpacking since tuples are inmutable
            exp = Experience(state0=s, action=a, reward=r, state1=observation, terminal1=t)  # s' is set to the current observation
            self.maj_experiences[last_index] = exp  # Overwrite last experience

        # None is placeholder for s'
        exp = Experience(state0=observation, action=action, reward=reward, state1=None, terminal1=terminal)
        if current_class:  # Add current experience in the rotating list, check if experience is minority or majority
            self.min_experiences[self.min_index] = exp
            self.min_size = min(self.min_size + 1, self.min_limit)
            self.min_index = (self.min_index + 1) % self.min_limit
        else:
            self.maj_experiences[self.maj_index] = exp
            self.maj_size = min(self.maj_size + 1, self.maj_limit)
            self.maj_index = (self.maj_index + 1) % self.maj_limit

        self.last_class = current_class

    @property
    def nb_entries(self):
        """Returns number of total observations."""
        return self.min_size + self.maj_size

    @staticmethod
    def calculate_batch_sizes(batch_size: int, minority_chance: float):
        """Calculates individual batch sizes w.r.t. the `minority_chance`."""
        batch_min = math.ceil(minority_chance * batch_size)
        batch_maj = batch_size - batch_min
        return batch_min, batch_maj

    def get_config(self):
        """Return configurations of PriorityMemory."""
        config = super(PriorityMemory, self).get_config()
        config['limit'] = self.limit
        config['minority_chance'] = self.minority_chance
        return config


if __name__ == "__main__":
    memory = PriorityMemory(limit=40, window_length=1)
    [memory.append([1], 1, 1, True) for i in range(60)]  # s, a, r, t
    [memory.append([0], 1, 0.5, False) for i in range(60)]
    # print(len(memory.min_experiences), len(memory.maj_experiences))
    print(*[x[0][0] for x in memory.sample(19)])
    print(PriorityMemory.calculate_batch_sizes(16, 0.125))
