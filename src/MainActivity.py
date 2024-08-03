import gymnasium as gym
# require numpy version before 1.26.4
from stable_baselines3 import SAC
from stable_baselines3.ppo.policies import MlpPolicy
from src.Novelty.NoveltyName import NoveltyName
from src.training.ModelTraining import trainNewModel, continueTrainingModel

# Set up environment for lunar lander
env = gym.make("LunarLander-v2", render_mode="human", continuous=True)

try:
    model = continueTrainingModel(NoveltyName.ORIGINAL)
except RuntimeError:
    model = trainNewModel(env, NoveltyName.ORIGINAL)
env = model.get_env()
observation = env.reset()

for _ in range(1000):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)

    if done:
        model = continueTrainingModel(NoveltyName.ORIGINAL)
        env = model.get_env()
        observation = env.reset()


env.close()
