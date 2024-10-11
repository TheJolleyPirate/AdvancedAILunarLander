import statistics
import gymnasium as gym
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from src.Util import adapt_observation


class ModelEvaluation:

    def evaluate(self, model: OffPolicyAlgorithm, env: gym.Env,  n_episodes: int = 100):
        rewards = []
        shape_trained = env.observation_space.shape[0]
        for _ in range(n_episodes):
            tmp, count_failed_eval = 0, 0
            observation, _ = env.reset()   # latest gym.Env always return `info`.
            observation = adapt_observation(observation, shape_trained)
            done = False
            while not done:
                action = model.predict(observation, deterministic=True)
                try:
                    observation, reward, done, truncated, _ = env.step(action)
                    observation = adapt_observation(observation, shape_trained)
                except RuntimeError:
                    count_failed_eval += 1
                    reward = 0
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
        return rewards