import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from src.novelty.NoveltyName import NoveltyName
from src.hyperparameters import LoadHyperparameters
from src.training.ModelAccess import saveModel, loadModel

num_episodes = 500_000


def trainNewModel(env: gym.Env, novelty_name: NoveltyName, inputNumEpisodes = None):
    print("Training new model for novelty: " + novelty_name.value)
    params = LoadHyperparameters.load("../admin/sac.yml")  # FIXME: not good practice of loading file.
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
    # By default model will reset # of timesteps, resulting 0 episodes of training
    # Hence save timesteps separately
    if inputNumEpisodes is None:
        model.N_TIMESTEPS = params.n_timesteps
    else:
        model.N_TIMESTEPS = inputNumEpisodes
    model.learn(total_timesteps=model.N_TIMESTEPS, progress_bar=True)
    saveModel(model, novelty_name)
    return model


def continueTrainingModel(env=None, novelty_name: NoveltyName = NoveltyName.ORIGINAL, inputNumEpisodes = num_episodes):
    model = loadModel(novelty_name)
    if env is not None:
         model.env = env
    print("Retraining model for novelty: " + novelty_name.value)
    model.learn(total_timesteps=inputNumEpisodes)
    saveModel(model, novelty_name)
    return model
