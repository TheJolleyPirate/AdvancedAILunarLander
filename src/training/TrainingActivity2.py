import argparse
from datetime import datetime, timedelta
from time import sleep
from src.novelty.NoveltyName import NoveltyName, noveltyList
from src.novelty.NoveltyDirector import NoveltyDirector
from src.exceptions.NoModelException import NoModelException
from src.training.ModelTraining import continueTrainingModel, trainNewModel


def training_activity(novel_name=NoveltyName.ORIGINAL, render = None, continuous = True, trainTime = 4):
    env = NoveltyDirector(novel_name).build_env(render, continuous)
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=trainTime)
    while datetime.now() < end_time:
        try:
            model = load()
        except NoModelException:
            trainNewModel(novel_name, novel_name)


def main():
    training_activity(novel_name=NoveltyName.SENSOR,
                      render=None,
                      continuous=True,
                      trainTime=4)

if __name__ == "__main__":
    main()
