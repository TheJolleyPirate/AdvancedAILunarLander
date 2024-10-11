import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.evaluation.ModelEvaluation import evaluate
from src.novelty.NoveltyDirector import NoveltyDirector
from src.novelty.NoveltyName import NoveltyName
from src.hyperparameters import LoadHyperparameters
from src.training.ModelAccess import save_model, loadModel, load_best_model

num_timesteps = 10_000

show_progress_bar = False
try:
    import tqdm
    import rich
    show_progress_bar = True
    print("Progress bar is enabled.")
except ImportError:
    print("Progress bar disabled. To enable it, install package `tqdm`, `rich`.")


def trainNewModel(env: gym.Env, novelty_name: NoveltyName):
    print("Training new model for novelty: " + novelty_name.value)
    
    # if not isinstance(env, Monitor):
    #     print("Wrapping env with Monitor for presentation.")
    #     env = Monitor(env)
    
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
    save_model(model, novelty_name)
    return model


def continueTrainingModel(env_novelty: NoveltyName = NoveltyName.ORIGINAL, 
                          model_novelty: NoveltyName = NoveltyName.ORIGINAL):
    env = NoveltyDirector(env_novelty).build_env()
    
    model = load_best_model(model_novelty)
    model.set_env(env)

    print("Retraining model for novelty: " + model_novelty.value)
    model = model.learn(total_timesteps=num_timesteps, progress_bar=show_progress_bar)
    save_model(model, model_novelty)
    return model


def continuous_with_decaying_learning(
        env_novelty: NoveltyName = NoveltyName.ORIGINAL,
        model_novelty: NoveltyName = NoveltyName.ORIGINAL,
        decaying_factor: float = 0.95):
    env = NoveltyDirector(env_novelty).build_env()
    num_episodes = 100

    prev_model = load_best_model(model_novelty)
    prev_mean = evaluate(prev_model, env, num_episodes)

    prev_model.set_env(env)

    print("Retraining model for novelty: " + model_novelty.value)
    current_model = prev_model.learn(total_timesteps=num_timesteps, progress_bar=show_progress_bar)
    current_mean = evaluate(current_model, env, num_episodes)

    if current_mean > prev_mean:
        current_model.learning_rate = current_model.learning_rate * decaying_factor
        save_model(current_model, model_novelty)
        return current_model

    else:
        print("Model not saved due to non-improving average reward.")
        return prev_model