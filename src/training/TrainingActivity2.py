import argparse
from collections import deque
from datetime import datetime, timedelta
from time import sleep

from src.hyperparameters import Hyperparameters
from src.novelty.NoveltyName import NoveltyName, noveltyList
from src.novelty.NoveltyDirector import NoveltyDirector
from src.exceptions.NoModelException import NoModelException
from src.training.ModelAccess import ModelAccess
from src.training.ModelTraining import trainNewModel, train_best, train_last, train_model
from src.training.TuningParameters import scale_learning_rate, scale_batch_size


def training_activity(novelty_name: NoveltyName = NoveltyName.ORIGINAL, render=None, continuous=True, trainTime=8):
    # setting up environment
    env = NoveltyDirector(novelty_name).build_env(render, continuous)
    model_access = ModelAccess(novelty_name)

    # dynamic updatable params
    params: Hyperparameters = None
    recent_success = []
    recent_files = []
    queue_size = 10
    prev_success = False

    start_time = datetime.now()
    end_time = start_time + timedelta(hours=trainTime)
    while datetime.now() < end_time:
        try:
            if prev_success:
                name, model = model_access.get_latest()
            else:
                name, model = model_access.get_best()
            result = train_model(env=env,
                                 prev_model=model,
                                 prev_filename=name,
                                 params=params,
                                 model_novelty=novelty_name)
            model_access.add_model(name, model)

            # Update records
            params = result.params
            prev_success = result.success
            recent_success.append(result.success)
            recent_files.append(result.filename)
            if len(recent_success) > queue_size:
                recent_success.pop(0)
                recent_files.pop(0)

            # Adjust hyper parameters
            # if the same file is training all the time
            count = [recent_files.count(s) for s in set(recent_files)]
            if len(count) == 1:
                # explore
                scale_learning_rate(params, 1.5)
                scale_batch_size(params, 0.25)
            elif any([x > queue_size / 2 for x in count]) or result.success:
                # exploit
                scale_learning_rate(params, 0.95)
                scale_batch_size(params, 2)
            else:
                # explore
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
