import gymnasium as gym
# require numpy version before 1.26.4
from stable_baselines3 import SAC
from stable_baselines3.ppo.policies import MlpPolicy
from src.Novelty.NoveltyName import NoveltyName
from src.training.ModelTraining import trainNewModel

# Set up environment for lunar lander
env = gym.make("LunarLander-v2", render_mode="human", continuous=True)
model = trainNewModel(env, NoveltyName.ORIGINAL)
env = model.env
# train model
observation = env.reset()

for _ in range(1000):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()

env.close()
