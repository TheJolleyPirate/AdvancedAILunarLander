import gymnasium
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from environments.atmosphereEnvironment import AtmosphereLunarLander
from environments.thrusterEnviro import FaultyThrusters
from src.Util import adapt_observation
from src.environments.TurbulenceEnv import TurbulenceEnv
from src.environments.gravityEnvironment import GravityLunarLander
from src.novelty.NoveltyName import NoveltyName
from src.novelty.limited_sensor.LimitedSensor import LimitedSensor
from src.training.ModelAccess import loadModel
import statistics


def main():
    n_eval_episodes = 100

    # Load latest trained model
    model = loadModel(NoveltyName.ORIGINAL)
    shape_trained = model.env.observation_space.shape[0]

    # load environment
    original_env = gymnasium.make("LunarLander-v2", render_mode=None, continuous=True)
    faulty_thruster_env = Monitor(FaultyThrusters(render_mode="human", continuous=True))
    atmosphere_env = AtmosphereLunarLander(render_mode="human", continuous=True)
    changing_gravity_env = Monitor(GravityLunarLander(render_mode="rgb_array", continuous=True))
    turbulence_env = Monitor(TurbulenceEnv(continuous=True))
    limited_sensor_env = LimitedSensor(render_mode="human")

    envs = [original_env, faulty_thruster_env, atmosphere_env, changing_gravity_env, turbulence_env, limited_sensor_env]
    env_names = ["Original", "Custom-Limited Sensor"]
    for i in range(len(envs)):
        print(f"Evaluating environment {env_names[i]}")
        evaluate(model, envs[i], 100)


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
