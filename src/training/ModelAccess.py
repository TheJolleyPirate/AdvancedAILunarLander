import os
import gymnasium as gym
from datetime import datetime
from typing import Optional

from stable_baselines3 import SAC
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from src.Novelty.NoveltyName import NoveltyName


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
        return Optional[OffPolicyAlgorithm]
    model_path = os.path.join(parent_folder, novelty_name.value)
    if not os.path.exists(model_path):
        return Optional[OffPolicyAlgorithm]
    filenames = os.listdir(model_path)
    if len(filenames) == 0:
        return Optional[OffPolicyAlgorithm]
    latest_filename = path=sorted(filenames, reverse=True)[0].removesuffix(".zip")
    p = os.path.join(parent_folder, novelty_name.value, latest_filename)
    return SAC.load(path=p, env=gym.make("LunarLander-v2", render_mode="human", continuous=True))
