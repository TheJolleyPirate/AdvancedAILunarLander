import argparse

from src.Util import adapt_observation
from src.Novelty.NoveltyDirector import NoveltyDirector
from src.Novelty.NoveltyName import NoveltyName, noveltyList
from src.exceptions.NoModelException import NoModelException
from src.training.ModelAccess import loadModel
import statistics

def runSingleNovelty(novelty, numEvalEpisodes, render, continuous):
    # load environment
    env = NoveltyDirector(novelty).build_env(render_mode=render, continuous=continuous)
    # load model
    try:
        # trying to get model matching novelty
        model = loadModel(novelty)
        usedModel = novelty.value
    except NoModelException:
        # if no model for current exception using original
        print(f"no model for {novelty} using default")
        usedNovelty = NoveltyName.ORIGINAL
        model = loadModel(usedNovelty)
        usedModel = usedNovelty.value
    # shape_trained = model.env.observation_space.shape[0]
    print(f"Evaluating environment {novelty.name} with model {usedModel}: ...")
    evaluate(model, env, numEvalEpisodes)
    print(f"Finish evaluation. \n")

def main(novelty: NoveltyName, render: str, continuous: bool, numEvalEpisodes: int):

    if novelty is None:
        for currentNovelty in NoveltyName:
            runSingleNovelty(currentNovelty, numEvalEpisodes, render, continuous)
    else:
        runSingleNovelty(novelty, numEvalEpisodes, render, continuous)

def evaluate(model, env, n_episodes: int = 100):
    rewards = []
    shape_trained = model.env.observation_space.shape[0]
    for _ in range(n_episodes):
        tmp = 0
        observation, _ = env.reset()
        done = False
        while not done:
            observation = adapt_observation(observation, shape_trained)
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, info = env.step(action)
            tmp += reward
        rewards.append(tmp)
    mean_reward = round(statistics.mean(rewards), 2)
    std_reward = round(statistics.stdev(rewards), 2)
    print(f"Number of episodes for evaluation: {n_episodes}")
    print(f"Mean reward: {mean_reward}")
    print(f"Standard deviation: {std_reward}")
    print(f"Min: {min(rewards)}")
    print(f"Max: {max(rewards)}")


if __name__ == '__main__':
    # USAGE:
    #       python MainActivity.py
    #       python MainActivity.py -n "NOVELTY"
    # or    python MainActivity.py -n "NOVELTY" -r "render_mode"
    # or    python MainActivity.py -nc
    # or    python MainActivity.py -n "NOVELTY" -nc
    # or    python MainActivity.py -n "NOVELTY" -r "render_mode" -nc

    parser = argparse.ArgumentParser(prog="MainActivity",
                                     description="the main activity to run the agents, can be run on "
                                                 "multiple different novelties")
    # CHANGE THIS IF YOU DON'T WANT TO USE THE CLI
    defaultNovelty = None
    allowedNovelties = noveltyList()
    allowedNovelties.append(None)
    parser.add_argument("-n", "--novelty", default=defaultNovelty, choices=allowedNovelties,
                        help="the novelty you want to run the agent on, leave blank or input \"all\"")
    # CHANGE THIS IF YOU DON'T WANT TO USE THE CLI
    defaultRender = "human"
    allowedRenders = [None, "human"]
    parser.add_argument("-r", "--render_mode", default=defaultRender, choices=allowedRenders,
                        help="the render mode you want to use")
    # CHANGE THIS IF YOU DON'T WANT TO USE THE CLI
    defaultContinous = True
    parser.add_argument("-nc", "--not_continuous", action="store_true", default=(not defaultContinous),
                        help="set this if you want the agent to be non-continuous (discrete)")
    # CHANGE THIS IF YOU DON'T WANT TO USE THE CLI
    defaultNumEvalEpisodes = 100
    parser.add_argument("-e", "--num_episodes", type=int, default=defaultNumEvalEpisodes,
                        help="number of evaluation episodes; default 100, must be at least 2")
    args = parser.parse_args()
    noveltyName = args.novelty
    runningNovelty = defaultNovelty
    for n in NoveltyName:
        if n.value == noveltyName:
            runningNovelty = n
        elif noveltyName == "all":
            runningNovelty = None
    runningRenderMode = args.render_mode
    runningContinuous = not args.not_continuous
    runningNumEpisodes = args.num_episodes
    if runningNumEpisodes < 2:
        raise ValueError("number of evaluation episodes must be at least 2")
    main(runningNovelty, runningRenderMode, runningContinuous, runningNumEpisodes)
