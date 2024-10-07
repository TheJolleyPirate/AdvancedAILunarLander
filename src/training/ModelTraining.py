import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.novelty.NoveltyDirector import NoveltyDirector
from src.novelty.NoveltyName import NoveltyName
from src.hyperparameters import LoadHyperparameters
from src.training.ModelAccess import saveModel, loadModel


num_timesteps = 5_000

show_progress_bar = False
try:
    import tqdm
    show_progress_bar = True
    print("Progress bar is enabled.")
except ImportError:
    print("Progress bar disabled. To enable it, install package `tqdm`. ")


def trainNewModel(env: gym.Env, novelty_name: NoveltyName):
    print("Training new model for novelty: " + novelty_name.value)
    
    if not isinstance(env, Monitor):
        print("Wrapping env with Monitor for presetation.")
        env = Monitor(env)
    
    params = LoadHyperparameters.load("../admin/sac.yml")  # FIXME: not good practice of loading file.
    
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
    saveModel(model, novelty_name)
    return model


def continueTrainingModel(env_novelty: NoveltyName = NoveltyName.ORIGINAL, 
                          model_novelty: NoveltyName = NoveltyName.ORIGINAL):
    env = NoveltyDirector(env_novelty).build_env()
    
    model = loadModel(model_novelty)
    model.set_env(env)

    print("Retraining model for novelty: " + model_novelty.value)
    model = model.learn(total_timesteps=num_episodes, progress_bar=show_progress_bar)
    saveModel(model, model_novelty)
    return model
