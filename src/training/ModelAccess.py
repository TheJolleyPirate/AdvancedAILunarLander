import os
import sys
from typing import List

import gymnasium as gym
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from src.evaluation import ModelEvaluation
from src.evaluation.ModelEvaluation import evaluate
from src.novelty.NoveltyDirector import NoveltyDirector
from src.novelty.NoveltyName import NoveltyName
from src.exceptions.NoModelException import NoModelException

date_format = "%Y%m%d-%H%M%S"
parent_folder = "models"


def save_model(model: OffPolicyAlgorithm, novelty_name: NoveltyName):
    # create directory
    model_path = _model_path(novelty_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    filename = datetime.now().strftime(date_format)
    model.save(os.path.join(model_path, filename))



def _model_path(novelty_name: NoveltyName) -> str:
    folder_path = os.path.join(os.getcwd(), "..", "..", parent_folder)
    # check non-empty folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # check sub-folder
    model_path = os.path.join(folder_path, novelty_name.value)
    return model_path


def _list_trained(novelty_name: NoveltyName) -> List[str]:
    model_path = _model_path(novelty_name)
    if not os.path.exists(model_path):
        raise NoModelException(novelty_name)

    # get all filenames in the folder. Assume all .zip files.
    filenames = os.listdir(model_path)
    if len(filenames) == 0:
        raise NoModelException(novelty_name)

    return [os.path.join(model_path, f.removesuffix(".zip")) for f in filenames]


def _load_model(novelty_name: NoveltyName, path_name: str, verbose: bool = False) -> OffPolicyAlgorithm:
    env = NoveltyDirector(novelty_name).build_env()

    if not verbose:
        # suppress printing the Wrapping message by stable-baselines3
        import contextlib
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            print("This will not be printed")
            loadedModel = SAC.load(path=path_name,
                                   env=env,
                                   custom_objects={'observation_space': env.observation_space,
                                                   'action_space': env.action_space})
    else:
        loadedModel = SAC.load(path=path_name,
                               env=env,
                               custom_objects={'observation_space': env.observation_space,
                                               'action_space': env.action_space})
    if verbose:
        print(f"Model loaded: {path_name}")
    return loadedModel


def load_latest(novelty_name: NoveltyName) -> OffPolicyAlgorithm:
    files = _list_trained(novelty_name)
    latest_name = sorted(files, reverse=True)[0]
    print(f"{novelty_name.value.upper()} latest model {latest_name} selected.")
    return _load_model(novelty_name, latest_name)


def load_best_model(novelty_name: NoveltyName) -> OffPolicyAlgorithm:
    files = _list_trained(novelty_name)
    target = 0
    best_mean = 1 - sys.maxsize

    # has to build env again, as stable-baselines3 wrapper not updated
    env = NoveltyDirector(novelty_name).build_env()
    # find index of best model.
    for i in range(len(files)):
        model = _load_model(novelty_name, files[i])
        mean = evaluate(model, env)
        if mean > best_mean:
            target = i
            best_mean = mean
    print(f"{novelty_name.value.upper()} model {files[target]} with mean of {round(best_mean, 2)} selected.")
    return _load_model(novelty_name, files[target])


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
    loadedModel = SAC.load(path=p, env=env, custom_objects={'observation_space': env.observation_space,
                                                            'action_space': env.action_space})
    print(f"{novelty_name.value.upper()} model loaded: {latest_filename}")
    return loadedModel
