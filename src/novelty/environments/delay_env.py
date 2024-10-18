""" Student Details

    Student Name: Kai Phan
    Student ID: u7779236
    Email: u7779236@anu.edu.au
    Date: 2024-08-24
"""

import gymnasium as gym
import numpy as np
from collections import deque

class ActionDelayWrapper(gym.Wrapper):
    def __init__(self, env, min_delay=5, max_delay=60):
        super(ActionDelayWrapper, self).__init__(env)
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.current_delay = np.random.randint(min_delay, max_delay)
        self.action_queue = deque(maxlen=self.current_delay)

        # Initialize the queue with a 'no-op' action or a random action
        self.no_op_action = self.action_space.sample()
        for _ in range(self.current_delay):
            self.action_queue.append(self.no_op_action)

    def step(self, action):
        self.action_queue.append(action)
        delayed_action = self.action_queue.popleft()

        obs, reward, terminated, truncated, info = self.env.step(delayed_action)

        # Update the delay and pass this info to the render method
        self.current_delay = np.random.randint(self.min_delay, self.max_delay)
        self.env.current_delay = self.current_delay  # Pass delay to the environment
        self.env.delayed_action = delayed_action  # Pass the delayed action

        self.action_queue = deque(list(self.action_queue), maxlen=self.current_delay)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_delay = np.random.randint(self.min_delay, self.max_delay)
        self.action_queue = deque([self.no_op_action] * self.current_delay, maxlen=self.current_delay)
        self.env.current_delay = self.current_delay
        return obs, info
