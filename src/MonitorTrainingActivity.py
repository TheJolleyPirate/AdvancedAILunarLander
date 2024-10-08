from typing import Optional
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from novelty.NoveltyDirector import NoveltyDirector
from stable_baselines3.common.callbacks import BaseCallback

from src.hyperparameters import LoadHyperparameters
from src.novelty.NoveltyName import NoveltyName
from src.training.ModelAccess import saveModel
from training import ModelAccess


def monitor_training(env_novelty=NoveltyName.ORIGINAL,
                      used_model_name: Optional[NoveltyName]=None,
                      num_episodes=5000):
    env = NoveltyDirector(env_novelty).build_env()

    if used_model_name is None:
        params = LoadHyperparameters.load()  # FIXME: not good practice of loading file.
        model = SAC(env=env,
                    batch_size=params.batch_size,
                    buffer_size=params.buffer_size,
                    ent_coef=params.ent_coef,
                    gamma=params.gamma,
                    gradient_steps=params.gradient_steps,
                    learning_rate=params.learning_rate,
                    policy=params.policy,
                    policy_kwargs=params.policy_kwargs,
                    verbose=1)
    else:
        model = ModelAccess.loadModel(used_model_name)
    model.set_env(env)
    rewards = []


    if used_model_name is None:
        printed_model_name = "original"
    else:
        printed_model_name = used_model_name.value
    print(f"Start training environment of {env_novelty.value} "
          f"with model of {printed_model_name} "
          f"for {num_episodes} episodes")

    for i in range(num_episodes):
        callback = EpisodeRewardCallback()
        model = model.learn(total_timesteps=10000, log_interval=1, callback=callback)
        rewards.append(callback.get_reward())
        if i in [int(num_episodes / 5 * j) for j in range(1, 11)]:
            print(f"Current progress: trained {i} episodes with reward {rewards[len(rewards) - 1]}")
    saveModel(model, env_novelty)
    print(rewards)
    plot(rewards, f"env-{env_novelty.value} model-{printed_model_name} ep-{num_episodes}")

def plot(array, filename):
    # Create a heatmap plot
    plt.plot(array)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards over Episodes")
    plt.savefig(f"{filename}.png")


class EpisodeRewardCallback(BaseCallback):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.reward = 0

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None and any(self.locals['dones']):
            self.reward = sum(self.locals['rewards'])
            return False    # train for only one episode 
        return True
    

    def _on_training_end(self) -> None:
        return None

    
    def get_reward(self):
        return self.reward

def main(target_novelty: Optional[NoveltyName] = None):
    if target_novelty is None:
        # train all of them
        for novelty in NoveltyName:
            if novelty == NoveltyName.ORIGINAL:
                continue
            monitor_training(env_novelty=novelty, num_episodes=5000)
    else:
        monitor_training(env_novelty=target_novelty, num_episodes=5000)

if __name__ == "__main__":
    main()