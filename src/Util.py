import numpy as np


def _trim_observation(observation, target_size):
    return observation[:target_size]


def _pad_observation(observation, target_size):
    return np.concatenate([observation, np.zeros(target_size - len(observation))])


def adapt_observation(observation, target_size):
    if len(observation) < target_size:
        return _pad_observation(observation, target_size)
    elif len(observation) > target_size:
        return _trim_observation(observation, target_size)
    return observation

