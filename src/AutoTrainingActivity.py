from datetime import datetime, timedelta
from time import sleep
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from novelty.NoveltyDirector import NoveltyDirector
from stable_baselines3.common.logger import configure

from src.novelty.NoveltyName import NoveltyName
from src.exceptions.NoModelException import NoModelException
from src.training.ModelTraining import continueTrainingModel, trainNewModel
from training import ModelAccess


def training_activity(novelty_name=NoveltyName.ORIGINAL):
    env_novelty = NoveltyName.ORIGINAL
    model_novelty = NoveltyName.ORIGINAL
    
    env = NoveltyDirector(env_novelty).build_env(render_mode="human")
    
    model = ModelAccess.loadModel(model_novelty)
    model.set_env(env)
    
    new_logger = configure('log/sac', ["stdout", "csv"])
    model.set_logger(new_logger)
    
    model = model.learn(total_timesteps=10000, log_interval=1)

        
        # training_results = model.env.get_episode_rewards()
        # plt.plot(np.arange(len(training_results)), training_results)
        # plt.title('Training Process: Episode Rewards')
        # plt.xlabel('Episode')
        # plt.ylabel('Reward')
        # plt.grid()
        # plt.show()
        # filename = datetime.now().strftime("%Y%m%d-%H%M%S")
        # plt.savefig(f"{filename}.png")


if __name__ == "__main__":
    training_activity()