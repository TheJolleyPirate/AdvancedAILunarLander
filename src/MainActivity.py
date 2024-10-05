import argparse

from src.Util import adapt_observation
from src.novelty.NoveltyDirector import NoveltyDirector
from src.novelty.NoveltyName import NoveltyName, noveltyList
from src.exceptions.NoModelException import NoModelException
from src.training.ModelAccess import loadModel
from gymnasium.envs.box2d.lunar_lander import LunarLander
import statistics

def runSingleNovelty(novelty, agent, numEvalEpisodes, render, continuous):
    # load environment
    env = NoveltyDirector(novelty).build_env(render_mode="rgb_array", continuous=continuous)
    
    # load model
    try:
        try:
            # trying to get model matching agent
            model = loadModel(agent)
            usedModel = agent.value
        except AttributeError:
            # agent is None type, or string using default (novelty agent
            model = loadModel(novelty)
            usedModel = novelty.value
    except NoModelException:
        # if no model for selected agent then using default
        print(f"no model for {agent} using default")
        usedNovelty = NoveltyName.ORIGINAL
        model = loadModel(usedNovelty)
        usedModel = usedNovelty.value

    print(f"Evaluating environment {novelty.name} with model {usedModel}: ...")
    evaluate(model, env, numEvalEpisodes)
    print(f"Finish evaluation. \n")

def main(novelty: NoveltyName, agent: NoveltyName, render: str, continuous: bool, numEvalEpisodes: int):

    if novelty is None:
        for currentNovelty in NoveltyName:
            runSingleNovelty(currentNovelty, agent, numEvalEpisodes, render, continuous)
    else:
        runSingleNovelty(novelty, agent, numEvalEpisodes, render, continuous)

def evaluate(model, env: LunarLander, n_episodes: int = 100):
    rewards = []
    shape_trained = model.env.observation_space.shape[0]
    for _ in range(n_episodes):
        tmp, count_failed_eval = 0, 0
        observation, _ = env.reset()
        done = False
        while not done:
            observation = adapt_observation(observation, shape_trained)
            action, _ = model.predict(observation, deterministic=True)
            try:
                observation, reward, done, truncated, _ = env.step(action)
            except RuntimeError:
                count_failed_eval += 1
            tmp += reward
        if count_failed_eval > 0:
            print(f"Failed to evaluate this novelty for {count_failed_eval} time(s).")
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
    parser.add_argument("-n", "--novelty", default=defaultNovelty,
                        help="the novelty you want to run the agent on, "
                             "leave blank or input \"all\" to run on all novelties")
    # CHANGE THIS IF YOU DON'T WANT TO USE THE CLI
    defaultagent = defaultNovelty
    parser.add_argument("-a", "--agent", default=defaultNovelty,
                        help="the agent you want to run on the novelty, "
                             "leave blank or input \"default\" to run on same as the novelty")
    # CHANGE THIS IF YOU DON'T WANT TO USE THE CLI
    defaultRender = None
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
    noveltyValue = args.novelty
    agentValue = args.agent
    runningNovelty = defaultNovelty
    runningAgent = defaultagent
    if noveltyValue != "all":
        for n in NoveltyName:
            if n.value == noveltyValue:
                runningNovelty = n
                break
    else:
        runningNovelty = None
    if agentValue != "default":
        for n in NoveltyName:
            if n.value == agentValue:
                runningAgent = n
                break
    else:
        runningAgent = None
    runningRenderMode = args.render_mode
    runningContinuous = not args.not_continuous
    runningNumEpisodes = args.num_episodes
    if runningNumEpisodes < 2:
        raise ValueError("number of evaluation episodes must be at least 2")
    main(runningNovelty, runningAgent, runningRenderMode, runningContinuous, runningNumEpisodes)
