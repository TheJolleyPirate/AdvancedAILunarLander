from stable_baselines3.common.evaluation import evaluate_policy
from src.Novelty.NoveltyName import NoveltyName
from src.training.ModelAccess import loadModel
import numpy as np
import statistics


def main():
    n_eval_episodes = 100

    # Load latest trained model
    model = loadModel(NoveltyName.ORIGINAL)
    env = model.get_env()

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"Mean reward for {n_eval_episodes}: {mean_reward}")
    print(f"Standard deviation of rewards: {std_reward}")

    for _ in range(n_eval_episodes):
        observation = env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
    env.close()


if __name__ == '__main__':
    main()
