import gymnasium as gym
# require numpy version before 1.26.4
from stable_baselines3 import SAC
from stable_baselines3.ppo.policies import MlpPolicy
from src.Novelty.NoveltyName import NoveltyName
from src.exceptions.NoModelException import NoModelException
from src.training.ModelTraining import trainNewModel, continueTrainingModel

# Set up environment for lunar lander
env = gym.make("LunarLander-v2", render_mode="human", continuous=True)
# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=int(2e5), progress_bar=True)
# env = model.get_env()
# observation = env.reset()

for _ in range(10):
    try:
        model = continueTrainingModel(NoveltyName.ORIGINAL)
    except NoModelException:
        model = trainNewModel(env, NoveltyName.ORIGINAL)

    env = model.get_env()
    observation = env.reset()


for _ in range(10):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)

    if done:
        model = continueTrainingModel(NoveltyName.ORIGINAL)
        env = model.get_env()
        observation = env.reset()

env.close()
