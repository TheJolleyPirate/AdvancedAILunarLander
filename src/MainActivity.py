from src.Util import adapt_observation
from src.novelty.NoveltyDirector import NoveltyDirector
from src.novelty.NoveltyName import NoveltyName
from src.training.ModelAccess import loadModel
import statistics


def main():
    n_eval_episodes = 100

    # Load latest trained model
    model = loadModel(NoveltyName.ORIGINAL)
    # shape_trained = model.env.observation_space.shape[0]

    # load environment
    for novelty in NoveltyName:
        env = NoveltyDirector(novelty).build_env(render_mode=None, continuous=True)
        print(f"Evaluating environment {novelty.name}: ...")
        evaluate(model, env, n_eval_episodes)
        print(f"Finish evaluation. \n")


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
    main()
