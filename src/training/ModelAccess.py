import os
from datetime import datetime
from typing import Optional

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from src.Novelty.NoveltyName import NoveltyName


date_format = "%Y%m%d-%H%M%S"
parent_folder = "../models/"


def saveModel(model: OffPolicyAlgorithm, novelty_name: NoveltyName):
    # create directory
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    model_path = os.path.join(parent_folder, novelty_name.name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    filename = datetime.now().strftime(date_format)
    model.save(os.path.join(model_path, filename))


def loadModel(novelty_name: NoveltyName):
    # check non-empty folder
    if not os.path.exists(parent_folder):
        return Optional[OffPolicyAlgorithm]
    model_path = os.path.join(parent_folder, novelty_name.name)
    if not os.path.exists(model_path):
        return Optional[OffPolicyAlgorithm]
    filenames = os.listdir(model_path)
    if len(filenames) == 0:
        return Optional[OffPolicyAlgorithm]
    return sorted(filenames, reverse=True)[0]
