import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from src.Novelty.NoveltyName import NoveltyName
from src.hyperparameters import LoadHyperparameters
from src.training.ModelAccess import saveModel, loadModel


def trainNewModel(env: gym.Env, novelty_name: NoveltyName):
    params = LoadHyperparameters.load("../admin/sac.yml")    # FIXME: not good practice of loading file.
    model = SAC(env=env,
                policy=params.policy,
                learning_rate=params.learning_rate,
                buffer_size=params.buffer_size,
                learning_starts=params.learning_starts,
                batch_size=params.batch_size,
                tau=params.tau,
                gamma=params.gamma,
                train_freq=params.train_freq,
                gradient_steps=params.gradient_steps,
                ent_coef=params.ent_coef,
                policy_kwargs=params.policy_kwargs)
    model.learn(total_timesteps=model.num_timesteps * 100_000)
    saveModel(model, NoveltyName.ORIGINAL)
    return model


def continueTrainingModel(novelty_name: NoveltyName):
    model = loadModel(novelty_name)
    model.learn(total_timesteps=model.num_timesteps * 100_000)
    saveModel(model, NoveltyName.ORIGINAL)
    return model
