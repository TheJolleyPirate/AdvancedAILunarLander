import os

import gymnasium as gym
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from src.Novelty.NoveltyName import NoveltyName
from src.exceptions.NoModelException import NoModelException

# Import custom lunar lander for novelties
from environments.atmosphereEnvironment import AtmosphereLunarLander

date_format = "%Y%m%d-%H%M%S"
parent_folder = "../models/"


def saveModel(model: OffPolicyAlgorithm, novelty_name: NoveltyName):
    # create directory
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    model_path = os.path.join(parent_folder, novelty_name.value)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    filename = datetime.now().strftime(date_format)
    model.save(os.path.join(model_path, filename))


def loadModel(novelty_name: NoveltyName) -> OffPolicyAlgorithm:
    # check non-empty folder
    if not os.path.exists(parent_folder):
        raise NoModelException(novelty_name)
    # check sub-folder
    model_path = os.path.join(parent_folder, novelty_name.value)
    if not os.path.exists(model_path):
        raise NoModelException(novelty_name)

    filenames = os.listdir(model_path)
    if len(filenames) == 0:
        raise NoModelException(novelty_name)
    latest_filename = sorted(filenames, reverse=True)[0].removesuffix(".zip")
    print(f"Model loaded: {latest_filename}")
    p = os.path.join(parent_folder, novelty_name.value, latest_filename)

    # if novelty_name == NoveltyName.ATMOSPHERE:
    #     env = AtmosphereLunarLander(render_mode="human", continuous=True)
    # else:
    #     env = gym.make("LunarLander-v2", render_mode="human", continuous=True)

    env = gym.make("LunarLander-v2", render_mode="human", continuous=True)

    return SAC.load(path=p, env=env, custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    # return SAC.load(path=p, env=gym.make("LunarLander-v2", render_mode="human", continuous=True))
