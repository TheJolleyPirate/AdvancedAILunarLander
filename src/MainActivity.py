from stable_baselines3.common.evaluation import evaluate_policy

from src.Util import adapt_observation
from src.novelty.NoveltyName import NoveltyName
from src.novelty.limited_sensor.LunarEnvironment import LunarEnvironment
from src.training.ModelAccess import loadModel


def main():
    n_eval_episodes = 100

    # Load latest trained model
    model = loadModel(NoveltyName.ORIGINAL)
    shape_trained = model.env.observation_space.shape[0]
    env = LunarEnvironment()

    # mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=n_eval_episodes)
    # print(f"Number of episodes for evaluation: {n_eval_episodes}")
    # print(f"Mean reward for {n_eval_episodes}: {mean_reward}")
    # print(f"Standard deviation of rewards: {std_reward}")

    n_presentation_episodes = 10
    for _ in range(n_presentation_episodes):
        observation, _ = env.reset()

        done = False
        while not done:
            observation = adapt_observation(observation, shape_trained)
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, info = env.step(action)
    env.close()


if __name__ == '__main__':
    main()
