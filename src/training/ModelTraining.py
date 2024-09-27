import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.novelty.NoveltyName import NoveltyName
from src.hyperparameters import LoadHyperparameters
from src.training.ModelAccess import saveModel, loadModel


num_episodes = 500_000


def trainNewModel(env: gym.Env, novelty_name: NoveltyName):
    print("Training new model for novelty: " + novelty_name.value)
    params = LoadHyperparameters.load("../admin/sac.yml")  # FIXME: not good practice of loading file.
    env = Monitor(DummyVecEnv[lambda: env])
    model = SAC(env=env,
                batch_size=params.batch_size,
                buffer_size=params.buffer_size,
                ent_coef=params.ent_coef,
                gamma=params.gamma,
                gradient_steps=params.gradient_steps,
                learning_rate=params.learning_rate,
                policy=params.policy,
                policy_kwargs=params.policy_kwargs,
                verbose=1)
    # By default model will reset # of timesteps, resulting 0 episodes of training
    # Hence save timesteps separately
    model.N_TIMESTEPS = params.n_timesteps

    model.learn(total_timesteps=model.N_TIMESTEPS, progress_bar=True)
    
    # training_results = env.get_episode_rewards()
    # plt.plot(np.arange(len(training_results)), training_results)
    # plt.title('Training Process: Episode Rewards')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.grid()
    # plt.show()

    saveModel(model, novelty_name)
    return model


def continueTrainingModel(env=None, novelty_name: NoveltyName = NoveltyName.ORIGINAL):
    model = loadModel(novelty_name)
    if env is not None:
        model.env = DummyVecEnv([lambda: env])
    print("Retraining model for novelty: " + novelty_name.value)
    model.learn(total_timesteps=num_episodes)
    saveModel(model, novelty_name)
    return model
