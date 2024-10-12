from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.evaluation.ModelEvaluation import evaluate
from src.hyperparameters.Hyperparameters import HyperParameters
from src.novelty.NoveltyDirector import NoveltyDirector
from src.novelty.NoveltyName import NoveltyName
from src.hyperparameters import LoadHyperparameters
from src.training.ModelAccess import save_model, loadModel, load_best_model, load_latest_model
from src.training.TrainingResult import TrainingResult
from src.training.TuningParameters import set_hyperparameters, get_hyperparameters

num_timesteps = 10_000

show_progress_bar = False
try:
    import tqdm
    import rich

    show_progress_bar = True
    print("Progress bar is enabled.")
except ImportError:
    print("Progress bar disabled. To enable it, install package `tqdm`, `rich`.")


def trainNewModel(env: gym.Env, novelty_name: NoveltyName, params: HyperParameters):
    print("Training new model for novelty: " + novelty_name.value)

    # if not isinstance(env, Monitor):
    #     print("Wrapping env with Monitor for presentation.")
    #     env = Monitor(env)

    model = SAC(env=env,
                batch_size=params.batch_size,
                buffer_size=params.buffer_size,
                ent_coef=params.ent_coef,
                gamma=params.gamma,
                gradient_steps=params.gradient_steps,
                learning_rate=params.learning_rate,
                policy=params.policy,
                policy_kwargs=params.policy_kwargs,
                verbose=0
                )

    # By default model will reset # of timesteps, resulting 0 episodes of training
    # Hence save timesteps separately

    model.N_TIMESTEPS = params.n_timesteps
    model.learn(total_timesteps=model.N_TIMESTEPS, progress_bar=True)
    filename = save_model(model, novelty_name)
    return TrainingResult(model, get_hyperparameters(model), True, filename)


def continueTrainingModel(env_novelty: NoveltyName = NoveltyName.ORIGINAL,
                          model_novelty: NoveltyName = NoveltyName.ORIGINAL):
    env = NoveltyDirector(env_novelty).build_env()

    model = load_best_model(model_novelty)
    model.set_env(env)

    print("Retraining model for novelty: " + model_novelty.value)
    model = model.learn(total_timesteps=num_timesteps, progress_bar=show_progress_bar)
    filename = save_model(model, model_novelty)
    return model


def train_last(
        env_novelty: NoveltyName = NoveltyName.ORIGINAL,
        model_novelty: NoveltyName = NoveltyName.ORIGINAL,
        params: HyperParameters = Optional[None]) -> TrainingResult:
    env = NoveltyDirector(env_novelty).build_env()
    prev_model, prev_filename = load_latest_model(model_novelty)
    num_episodes = 100
    return train_model(env, prev_model, prev_filename, params, model_novelty, num_episodes)


def train_best(
        env_novelty: NoveltyName = NoveltyName.ORIGINAL,
        model_novelty: NoveltyName = NoveltyName.ORIGINAL,
        params: HyperParameters = Optional[None]) -> TrainingResult:
    env = NoveltyDirector(env_novelty).build_env()
    prev_model, prev_filename = load_best_model(model_novelty)
    num_episodes = 100
    return train_model(env, prev_model, prev_filename, params, model_novelty, num_episodes)


def train_model(env,
                prev_model,
                prev_filename,
                params,
                model_novelty,
                num_episodes: int = 100,
                prev_mean: Optional[int] = None):
    prev_model.set_env(env)
    if prev_mean is None:
        prev_mean = evaluate(prev_model, env, num_episodes)

    # load hyperparameters
    if params is not None:
        set_hyperparameters(params, prev_model)
    current_model = prev_model.learn(total_timesteps=num_timesteps, progress_bar=show_progress_bar)
    current_mean = evaluate(current_model, env, num_episodes)

    if current_mean > prev_mean:
        filename = save_model(current_model, model_novelty)
        return TrainingResult(current_model, get_hyperparameters(current_model), True, filename)

    else:
        print("Model not saved due to non-improving average reward. Returning previous model.")
        return TrainingResult(prev_model, get_hyperparameters(prev_model), False, prev_filename)
