import gymnasium as gym
# require numpy version before 1.26.4
from stable_baselines3 import SAC
from stable_baselines3.ppo.policies import MlpPolicy
import yaml

# Set up environment for lunar lander
env = gym.make("LunarLander-v2", render_mode="human", continuous=True)

# load hyperparameters from stable-baseline3 zoo
try:
    hyperparams = yaml.safe_load(open("../admin/sac.yml"))
    model = SAC("MlpPolicy", env, verbose=1,
                learning_rate=hyperparams["learning_rate"],
                buffer_size=hyperparams["buffer_size"],
                )

except RuntimeError:
    hyperparams = {}

# train model
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
