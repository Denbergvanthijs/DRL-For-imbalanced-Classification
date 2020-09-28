import itertools
import math
import random
from collections import deque

from rl.memory import Experience, Memory


class PriorityMemory(Memory):
    """
    Samples minority and majority samples.
    Partly based on keras-rl's SequentialMemory.
    """

    def __init__(self, limit, minority_chance: float = 0.5, **kwargs):
        super(PriorityMemory, self).__init__(**kwargs)

        if isinstance(limit, (tuple, int)):
            # TODO: Add checking for tuple length.
            self.limit = limit
        else:
            raise TypeError("limit must be a tuple of length 2 or an integer.")

        if isinstance(self.limit, int):
            self.minority_experiences = deque(maxlen=limit//2)
            self.majority_experiences = deque(maxlen=limit//2)
        else:
            self.minority_experiences = deque(maxlen=limit[0])
            self.majority_experiences = deque(maxlen=limit[1])

        self.minority_chance = minority_chance
        self.last_class = None

    def sample(self, batch_size: int, batch_idxs=None):
        """
        Samples 50/50 minority and majority samples of size `batch_size`.
        Slicing of deque is based on: https://stackoverflow.com/a/7064775/10603874
        """
        if self.nb_entries <= batch_size:
            # Since last state has no next state, batch_size can't be equal to memory size
            raise ValueError("Sample larger or equal to memory")

        batch_min, batch_maj = self.calculate_batch_sizes(batch_size, self.minority_chance)
        minibatch = []

        len_min, len_maj = self.useful_length  # Amount of usable experiences per class
        if len_min < batch_min:
            batch_maj += batch_min - len_min
            batch_min = len_min
        if len_maj < batch_maj:
            batch_min += batch_maj - len_maj
            batch_maj = len_maj

        assert batch_size == batch_min + batch_maj, "This should not happen"

        # If any batch was not the same as the corresponding useful length, slices will be the same as original deque's
        min_slice = tuple(itertools.islice(self.minority_experiences, 0, batch_min))
        minibatch += random.sample(min_slice, batch_min)
        maj_slice = tuple(itertools.islice(self.majority_experiences, 0, batch_maj))
        minibatch += random.sample(maj_slice, batch_maj)

        random.shuffle(minibatch)
        return minibatch

    def append(self, observation, action, reward, terminal, training=True):
        """Append experience to memory. Update the next state of the last appended experience."""
        super(PriorityMemory, self).append(observation, action, reward, terminal, training=training)

        if not training:  # If testing
            return

        current_class = True if reward in (-1, 1) else False

        if self.last_class is not None:  # If there's ever been an experience
            if self.last_class:  # If the last experience was a minority, set s' to the current state
                s, a, r, _, t = self.minority_experiences.pop()
                exp = Experience(state0=s, action=a, reward=r, state1=observation, terminal1=t)  # s' is set to the current observation
                self.minority_experiences.append(exp)
            else:
                s, a, r, _, t = self.majority_experiences.pop()
                exp = Experience(state0=s, action=a, reward=r, state1=observation, terminal1=t)  # s' is set to the current observation
                self.majority_experiences.append(exp)

        # None is placeholder for s'
        exp = Experience(state0=observation, action=action, reward=reward, state1=None, terminal1=terminal)
        if current_class:  # Add current experience in the deque, check if experience is minority or majority
            self.minority_experiences.append(exp)
        else:
            self.majority_experiences.append(exp)

        self.last_class = current_class  # True if current exp is minority, False if not

    @property
    def nb_entries(self):
        """Returns number of total observations."""
        return len(self.minority_experiences) + len(self.majority_experiences)

    @property
    def useful_length(self):
        """Returns tuple with amount of useful elements per class."""
        if self.last_class is None:  # If no experience has been added yet
            return 0, 0

        if self.last_class:
            len_min = len(self.minority_experiences) - 1  # Last experience can't be used since it has no proper s'
            len_maj = len(self.majority_experiences)
        else:
            len_min = len(self.minority_experiences)
            len_maj = len(self.majority_experiences) - 1  # Last experience can't be used since it has no proper s'

        return len_min, len_maj

    @staticmethod
    def calculate_batch_sizes(batch_size: int, minority_chance: float):
        batch_min = math.ceil(minority_chance * batch_size)
        batch_maj = batch_size - batch_min
        return batch_min, batch_maj

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(PriorityMemory, self).get_config()
        config['limit'] = self.limit
        config['minority_chance'] = self.minority_chance
        return config


if __name__ == "__main__":
    memory = PriorityMemory(limit=1_000, window_length=1)
    [memory.append([1], 1, 1, True) for i in range(5)]  # s, a, r, t
    [memory.append([0], 1, 0.5, False) for i in range(5)]
    print(len(memory.minority_experiences), len(memory.majority_experiences))
    print(*[x[0][0] for x in memory.sample(4)])
    print(PriorityMemory.calculate_batch_sizes(64, 0.01))
