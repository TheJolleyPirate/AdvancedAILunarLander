import gymnasium
from stable_baselines3.common.evaluation import evaluate_policy
from src.Novelty.NoveltyName import NoveltyName
from src.training.ModelAccess import loadModel

from environments.gravityEnvironment import GravityLunarLander
from stable_baselines3.common.monitor import Monitor

def main():
    n_eval_episodes = 100

    # Load latest trained model
    model = loadModel(NoveltyName.ORIGINAL)
    env = Monitor(GravityLunarLander(render_mode="human", continuous=True))
    # env = gymnasium.make("LunarLander-v2", render_mode="rgb_array", continuous=True)

    mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=n_eval_episodes)
    print(f"Number of episodes for evaluation: {n_eval_episodes}")
    print(f"Mean reward for {n_eval_episodes}: {mean_reward}")
    print(f"Standard deviation of rewards: {std_reward}")

    n_presentation_episodes = 10
    for _ in range(n_presentation_episodes):
        observation = env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
    env.close()


if __name__ == '__main__':
    main()
