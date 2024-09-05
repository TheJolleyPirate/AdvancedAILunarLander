from datetime import datetime, timedelta
from time import sleep
import gymnasium as gym
from src.novelty.NoveltyName import NoveltyName
from src.exceptions.NoModelException import NoModelException
from src.novelty.limited_sensor.LimitedSensor import LimitedSensor
from src.training.ModelTraining import continueTrainingModel, trainNewModel


def training_activity(novel_name=NoveltyName.ORIGINAL):
    env = LimitedSensor(render_mode=None)
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=4)
    while datetime.now() < end_time:
        try:
            continueTrainingModel(env=env, novelty_name=NoveltyName.SENSOR)
            sleep(20)
        except NoModelException:
            trainNewModel(env=env, novelty_name=NoveltyName.SENSOR)


if __name__ == "__main__":
    training_activity()
