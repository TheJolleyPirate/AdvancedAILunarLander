from datetime import datetime, timedelta
from time import sleep
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from novelty.NoveltyDirector import NoveltyDirector
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

from src.novelty.NoveltyName import NoveltyName
from src.exceptions.NoModelException import NoModelException
from src.training.ModelTraining import continueTrainingModel, trainNewModel
from training import ModelAccess


def training_activity(novelty_name=NoveltyName.ORIGINAL):
    env_novelty = NoveltyName.ORIGINAL
    model_novelty = NoveltyName.ORIGINAL
    
    env = NoveltyDirector(env_novelty).build_env()
    
    model = ModelAccess.loadModel(model_novelty)
    model.set_env(env)

    new_logger = configure('log/sac', ["stdout", "csv"])
    model.set_logger(new_logger)
    
    model = model.learn(total_timesteps=10000, log_interval=1, callback=EpisodeRewardCallback())


class EpisodeRewardCallback(BaseCallback):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.reward = 0

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None and any(self.locals['dones']):
            self.reward = sum(self.locals['rewards'])
            return False    # train for only one episode 
        return True
    

    def _on_training_end(self) -> None:
        print(f"Episode finished with reward: {self.reward}")

    
    def get_reward(self):
        return self.reward



if __name__ == "__main__":
    training_activity()