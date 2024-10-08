from datetime import datetime, timedelta
from time import sleep
from typing import Optional

from src.novelty.NoveltyDirector import NoveltyDirector
from src.novelty.NoveltyName import NoveltyName
from src.exceptions.NoModelException import NoModelException
from src.training.ModelTraining import continueTrainingModel, trainNewModel


def training_activity(env_novelty=NoveltyName.SENSOR, model_novelty=NoveltyName.SENSOR,
                      pretrained_novelty: Optional[NoveltyName] = NoveltyName.ORIGINAL):
    # Set maximum training time frame
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=30)

    if pretrained_novelty is not None:
        continueTrainingModel(env_novelty=env_novelty, model_novelty = pretrained_novelty)

    env = NoveltyDirector(env_novelty).build_env()
    while datetime.now() < end_time:
        try:
            continueTrainingModel(env_novelty=env_novelty, model_novelty=model_novelty)
            sleep(5)
        except NoModelException:
            trainNewModel(env=env, novelty_name=env_novelty)


if __name__ == "__main__":
    training_activity(env_novelty=NoveltyName.SENSOR, model_novelty=NoveltyName.SENSOR, pretrained_novelty=None)

