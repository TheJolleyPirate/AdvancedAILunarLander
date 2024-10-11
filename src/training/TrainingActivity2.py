import argparse
from datetime import datetime, timedelta
from time import sleep
from src.novelty.NoveltyName import NoveltyName, noveltyList
from src.novelty.NoveltyDirector import NoveltyDirector
from src.exceptions.NoModelException import NoModelException
from src.training.ModelTraining import continueTrainingModel, trainNewModel, continuous_with_decaying_learning


def training_activity(novelty_name: NoveltyName = NoveltyName.ORIGINAL, render=None, continuous=True, trainTime=4):
    env = NoveltyDirector(novelty_name).build_env(render, continuous)
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=trainTime)

    while datetime.now() < end_time:
        try:
            model = continuous_with_decaying_learning(novelty_name, novelty_name)
            model.learning_rate = 0.01
            sleep(20)
        except NoModelException:
            trainNewModel(env, novelty_name)


def main():
    training_activity(novelty_name=NoveltyName.SENSOR,
                      render=None,
                      continuous=True,
                      trainTime=4)


if __name__ == "__main__":
    main()
