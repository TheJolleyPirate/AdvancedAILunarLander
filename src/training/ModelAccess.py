import os
import sys
from typing import List

import gymnasium as gym
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from src.evaluation import ModelEvaluation
from src.evaluation.EvaluationManager import EvaluationManager
from src.evaluation.ModelEvaluation import evaluate
from src.hyperparameters.Hyperparameters import HyperParameters
from src.novelty.NoveltyDirector import NoveltyDirector
from src.novelty.NoveltyName import NoveltyName
from src.exceptions.NoModelException import NoModelException

date_format = "%Y%m%d-%H%M%S"
parent_folder = "models"


def _model_path(novelty_name: NoveltyName) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, "..", "..", parent_folder)
    # check non-empty folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # check sub-folder
    model_path = os.path.join(folder_path, novelty_name.value)
    return model_path


def save_model(model: OffPolicyAlgorithm, novelty_name: NoveltyName) -> str:
    # create directory
    model_path = _model_path(novelty_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    filename = datetime.now().strftime(date_format)
    complete_filename = os.path.join(model_path, filename)
    model.save(complete_filename)
    return complete_filename


def _list_trained(novelty_name: NoveltyName) -> List[str]:
    model_path = _model_path(novelty_name)
    if not os.path.exists(model_path):
        raise NoModelException(novelty_name)

    # get all filenames in the folder. Assume all .zip files.
    filenames = os.listdir(model_path)
    if len(filenames) == 0:
        raise NoModelException(novelty_name)

    return [os.path.join(model_path, f.removesuffix(".zip")) for f in filenames]


def _load_model(novelty_name: NoveltyName,
                path_name: str,
                verbose: bool = False,
                params: HyperParameters = HyperParameters()) -> OffPolicyAlgorithm:
    env = NoveltyDirector(novelty_name).build_env()

    if not verbose:
        # suppress printing the Wrapping message by stable-baselines3
        import contextlib
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            print("This will not be printed")
            loadedModel = SAC.load(
                path=path_name,
                env=env,
                custom_objects={'observation_space': env.observation_space,
                                'action_space': env.action_space},
                verbose=1,
                num_timesteps=params.n_timesteps,
                policy=params.policy,
                learning_rate=params.learning_rate,
                buffer_size=params.buffer_size,
                batch_size=params.batch_size,
                gradient_steps=params.gradient_steps,
                gamma=params.gamma,
                tau=params.tau,
                learning_starts=params.learning_starts,
                use_sde=params.use_sde
            )

    if verbose:
        print(f"Model loaded: {path_name}")
    return loadedModel


def load_latest_model(novelty_name: NoveltyName) -> (OffPolicyAlgorithm, str):
    files = _list_trained(novelty_name)
    latest_name = sorted(files, reverse=True)[0]
    print(f"{novelty_name.value.upper()} latest model {latest_name} selected.")
    return _load_model(novelty_name, latest_name), latest_name


def load_best_model(novelty_name: NoveltyName) -> (OffPolicyAlgorithm, str):
    files = _list_trained(novelty_name)
    target = 0
    best_mean = 1 - sys.maxsize

    # has to build env again, as stable-baselines3 wrapper not updated
    env = NoveltyDirector(novelty_name).build_env()
    # find index of best model.
    for i in range(len(files)):
        model = _load_model(novelty_name, files[i])
        mean = evaluate(model, env).mean
        if mean > best_mean:
            target = i
            best_mean = mean
    print(f"{novelty_name.value.upper()} model {files[target]} with mean of {round(best_mean, 2)} selected.")
    return _load_model(novelty_name, files[target]), files[target]


def loadModel(novelty_name: NoveltyName) -> OffPolicyAlgorithm:
    # check non-empty folder
    if not os.path.exists(parent_folder):
        print(os.listdir())
        raise NoModelException(novelty_name)
    # check sub-folder
    model_path = os.path.join(os.getcwd(), "..", parent_folder, novelty_name.value)
    if not os.path.exists(model_path):
        raise NoModelException(novelty_name)

    filenames = os.listdir(model_path)
    if len(filenames) == 0:
        raise NoModelException(novelty_name)

    latest_filename = sorted(filenames, reverse=True)[0].removesuffix(".zip")
    p = os.path.join(model_path, latest_filename)
    env = NoveltyDirector(novelty_name).build_env()
    loadedModel = SAC.load(path=p,
                           env=env,
                           custom_objects={'observation_space': env.observation_space,
                                           'action_space': env.action_space},
                           verbose=1)
    print(f"{novelty_name.value.upper()} model loaded: {latest_filename}")
    return loadedModel


class ModelAccess:

    def __init__(self, novelty_name: NoveltyName, num_episodes: int =100):
        self.novelty_name = novelty_name
        env = NoveltyDirector(novelty_name).build_env()
        self.evaluation = EvaluationManager(env, novelty_name, num_episodes)
        self._load_models()

    def __del__(self):
        del self.evaluation

    def _load_models(self):
        try:
            files = _list_trained(novelty_name=self.novelty_name)
            print(f"Adding {len(files)} file(s) for evaluation: ", end="")
            for f in files:
                model = _load_model(self.novelty_name, f)
                self.evaluation.add_model(f, model)
                print("#", end="")
            print("\nEvaluation complete for ModelAccess.")
        except NoModelException:
            pass

    def load_latest_model(self, params: HyperParameters = HyperParameters()) -> (str, OffPolicyAlgorithm):
        filename = self.get_latest_name()
        return filename, _load_model(self.novelty_name, filename, False, params)

    def load_best_model(self, params: HyperParameters = HyperParameters()) -> (str, OffPolicyAlgorithm):
        filename = self.get_best_name()
        if filename is None or filename == "":
            raise NoModelException(self.novelty_name)
        return filename, _load_model(self.novelty_name, filename, False, params)

    def get_best_name(self):
        return self.evaluation.get_best_name()

    def get_latest_name(self):
        return self.evaluation.get_latest_name()

    def add_model(self, name, model):
        return self.evaluation.add_model(name, model)

    def has_model(self):
        return self.evaluation.has_model()

    def get_mean_reward(self, filename):
        return self.evaluation.get_mean_reward(filename)

    def get_performance(self, filename):
        return self.evaluation.performance[filename]