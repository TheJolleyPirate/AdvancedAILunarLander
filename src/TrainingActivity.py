import argparse
import random
from datetime import datetime, timedelta
from time import sleep

from stable_baselines3.common.vec_env import DummyVecEnv

from src.novelty.NoveltyName import NoveltyName
from src.novelty.NoveltyDirector import NoveltyDirector
from src.exceptions.NoModelException import NoModelException
from src.training.ModelTraining import continueTrainingModel, trainNewModel
from src.MemoryWrapper import MemoryWrapper


def training_activity(novel_name=NoveltyName.ORIGINAL, render = None, continuous = True, trainTime = 4):
    if novel_name != NoveltyName.MEMORY:
        env = NoveltyDirector(novel_name).build_env(render, continuous)
    else:
        env = None
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=trainTime)
    toTrain = [NoveltyName.ORIGINAL, NoveltyName.ATMOSPHERE, NoveltyName.GRAVITY,
               NoveltyName.SENSOR, NoveltyName.THRUSTER, NoveltyName.TURBULENCE]
    while datetime.now() < end_time:
        if novel_name == NoveltyName.MEMORY:
            index = random.randint(0,5)
            currentNovelty = toTrain[index]
            enviromentType = NoveltyDirector(currentNovelty)
            print(f"currently training {currentNovelty.value}")
            env = DummyVecEnv([lambda: MemoryWrapper(enviromentType.build_env(render, continuous))])
        try:
            continueTrainingModel(env=env, novelty_name=novel_name, inputNumEpisodes=2)
            sleep(20)
        except NoModelException:
            trainNewModel(env=env, novelty_name=novel_name, inputNumEpisodes=2)


if __name__ == "__main__":
    #USAGE:
    #       python TrainingActivity.py
    #       python TrainingActivity.py -n "NOVELTY"
    # or    python TrainingActivity.py -n "NOVELTY" -r "render_mode"
    # or    python TrainingActivity.py -nc
    # or    python TrainingActivity.py -n "NOVELTY" -nc
    # or    python TrainingActivity.py -n "NOVELTY" -r "render_mode" -nc

    parser = argparse.ArgumentParser(prog="TrainingActivity",
                                     description="the training activity to train the agents")
    # CHANGE THIS IF YOU DON'T WANT TO USE THE CLI
    defaultNovelty = NoveltyName.MEMORY
    parser.add_argument("-n", "--novelty", default=defaultNovelty,
                        help="the novelty you want to train the agent on")
    # CHANGE THIS IF YOU DON'T WANT TO USE THE CLI
    defaultRender = None
    allowedRenders = [None, "human"]
    parser.add_argument("-r", "--render_mode", default=defaultRender, choices=allowedRenders,
                        help="the render mode you want to use")
    # CHANGE THIS IF YOU DON'T WANT TO USE THE CLI
    defaultContinous = True
    parser.add_argument("-nc", "--not_continuous", action="store_true", default=(not defaultContinous),
                        help="set this if you want the agent to be non-continuous (discrete)")
    parser.add_argument("-t", "--training_time", type=int, default=4,
                        help="the amount of time to train the agent for; default 4")
    args = parser.parse_args()
    noveltyName = args.novelty
    trainingNovelty = defaultNovelty
    for n in NoveltyName:
        if n.value == noveltyName:
            trainingNovelty = n
    trainingRenderMode = args.render_mode
    trainingContinuous = not args.not_continuous
    trainingTime = args.training_time
    training_activity(trainingNovelty, trainingRenderMode, trainingContinuous, trainingTime)
