import gymnasium as gym
# require numpy version before 1.26.4
from stable_baselines3 import SAC
from stable_baselines3.ppo.policies import MlpPolicy
import yaml

from src.hyperparameters import LoadHyperparameters

# Set up environment for lunar lander
env = gym.make("LunarLander-v2", render_mode="human", continuous=True)

# load hyperparameters from stable-baseline3 zoo
parameters_file = "../admin/sac.yml"
params = LoadHyperparameters.load(parameters_file)

# train model
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
