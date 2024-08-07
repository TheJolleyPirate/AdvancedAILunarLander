from datetime import datetime, timedelta
from time import sleep
import gymnasium as gym
from src.Novelty.NoveltyName import NoveltyName
from src.exceptions.NoModelException import NoModelException
from src.training.ModelTraining import continueTrainingModel, trainNewModel


def training_activity(novel_name=NoveltyName.ORIGINAL):
    env = gym.make("LunarLander-v2", render_mode="human", continuous=True)
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=12)
    while datetime.now() < end_time:
        try:
            model = continueTrainingModel(NoveltyName.ORIGINAL)
            sleep(20)
        except NoModelException:
            model = trainNewModel(env, NoveltyName.ORIGINAL)


if __name__ == "__main__":
    training_activity()
