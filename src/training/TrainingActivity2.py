import argparse
from datetime import datetime, timedelta
from time import sleep

from src.hyperparameters import Hyperparameters
from src.novelty.NoveltyName import NoveltyName, noveltyList
from src.novelty.NoveltyDirector import NoveltyDirector
from src.exceptions.NoModelException import NoModelException
from src.training.ModelTraining import trainNewModel, train_dynamic_params
from src.training.TuningParameters import scale_learning_rate, scale_batch_size


def training_activity(novelty_name: NoveltyName = NoveltyName.ORIGINAL, render=None, continuous=True, trainTime=4):
    env = NoveltyDirector(novelty_name).build_env(render, continuous)
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=trainTime)
    params: Hyperparameters = None
    while datetime.now() < end_time:
        try:
            result = train_dynamic_params(novelty_name, novelty_name, params)
            params = result.params
            if result.success:
                scale_learning_rate(params, 0.95)
                scale_batch_size(params, 2)
            else:
                scale_learning_rate(params, 1.5)
                scale_batch_size(params, 0.25)

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
