import gymnasium as gym
import numpy as np
from math import sqrt


class DustVisibilityWrapper(gym.Wrapper):
    """
    A custom environment wrapper that simulates reduced visibility due to dust, with the effect starting at a certain
    height and increasing as the lander gets closer to the surface.

    Parameters:
    env: gym.Env
        The base environment to be wrapped.

    dust_start_height: float
        The height at which dust starts to affect visibility. This value should be between 0 and 1.

    max_mask_probability: float
        The maximum probability that the vision sensor fails when the lander is close to the surface.
        This value should be between 0 and 1.

    mask_value: float
        The value used to simulate the vision sensor fails (position_x and position_y) in the observation space.
    """

    def __init__(self, env, dust_start_height=0.8, max_mask_probability=0.8, max_noise_level=0.3, min_noise_level=0.1):
        super().__init__(env)
        self.dust_start_height = dust_start_height
        self.max_mask_probability = max_mask_probability
        self.max_noise_level = max_noise_level
        self.min_noise_level = min_noise_level

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        height = observation[1]

        # Determine the probability of masking based on the current height
        if height < self.dust_start_height:
            mask_probability = self._calculate_mask_probability(height)
            if np.random.uniform(0, 1) < mask_probability:
                noise_level = self._calculate_dynamic_noise_level(height)
                observation = self._apply_position_mask(observation, noise_level)

        return observation, reward, terminated, truncated, info

    def _calculate_mask_probability(self, height):
        # Calculate the masking probability based on the height
        return sqrt(self.max_mask_probability * (self.dust_start_height - height) / self.dust_start_height)

    def _calculate_dynamic_noise_level(self, height):
        # Calculate the noise level based on the height, increasing as the lander gets closer to the surface
        noise_level = self.min_noise_level + (self.max_noise_level - self.min_noise_level) * (
                    self.dust_start_height - height) / self.dust_start_height
        return sqrt(noise_level)

    def _apply_position_mask(self, observation, noise_level):
        # Mask the position elements (position_x and position_y) in the observation
        observation[0] += np.random.uniform(-noise_level, noise_level)
        observation[1] += np.random.uniform(-noise_level, noise_level)
        return observation



