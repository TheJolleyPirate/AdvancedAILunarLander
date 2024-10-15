import statistics
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from src.Util import adapt_observation
from gymnasium.envs.box2d.lunar_lander import LunarLander


def evaluate(model: OffPolicyAlgorithm, env: LunarLander, n_episodes: int = 100, verbose: bool = False):
    if n_episodes <= 0:
        return 0
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
            except RuntimeError as e:
                count_failed_eval += 1
                reward = 0
                print(e)
            tmp += reward
        if count_failed_eval > 0:
            print(f"Failed to evaluate this novelty for {count_failed_eval} time(s).")
        rewards.append(tmp)
    mean_reward = round(statistics.mean(rewards), 2)
    std_reward = round(statistics.stdev(rewards), 2)

    if verbose:
        print(f"Number of episodes for evaluation: {n_episodes}")
        print(f"Mean reward: {mean_reward}")
        print(f"Standard deviation: {std_reward}")
        print(f"Min: {min(rewards)}")
        print(f"Max: {max(rewards)}")

    return mean_reward, std_reward, min(rewards), max(rewards)
