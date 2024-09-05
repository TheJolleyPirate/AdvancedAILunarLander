from stable_baselines3.common.evaluation import evaluate_policy
from src.Novelty.NoveltyName import NoveltyName
from src.training.ModelAccess import loadModel
from environments.TurbulenceEnv import TurbulenceEnv
from stable_baselines3.common.monitor import Monitor


def main():
    n_eval_episodes = 100

    # Load latest trained model
    model = loadModel(NoveltyName.ORIGINAL)
    # model = loadModel(NoveltyName.ATMOSPHERE)
    # env = model.get_env()
    env = Monitor(FaultyThrusters(render_mode="human", continuous=True))
    model.set_env(env)
    env = Monitor(GravityLunarLander(render_mode="rgb_array", continuous=True))
    # env = gymnasium.make("LunarLander-v2", render_mode="rgb_array", continuous=True)
    # TODO: should load from NoveltyName model
    env = Monitor(TurbulenceEnv( continuous=True))
    model.set_env(env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"Mean reward for {n_eval_episodes}: {mean_reward}")
    print(f"Standard deviation of rewards: {std_reward}")

    for _ in range(n_eval_episodes):
        observation, _ = env.reset()
        # observation = env.reset() # (Without Monitor Wrapper / custom env)
        done = False
        while not done:
            action, _ = model.predict(observation=observation, deterministic=True)
            observation, reward, done, info, _ = env.step(action)
            # observation, reward, done, info = env.step(action) # (Without Monitor Wrapper / custom env)
    env.close()


if __name__ == '__main__':
    main()
