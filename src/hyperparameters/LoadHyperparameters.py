import yaml

from src.hyperparameters.Hyperparameters import HyperParameters


def load(filename: str):
    try:
        with open(filename) as file:
            # FIXME consider using package to load dictionary

            params_dict = yaml.safe_load(file)["LunarLanderContinuous-v2"]
            params = HyperParameters()
            params.n_timesteps = params_dict["n_timesteps"]
            params.policy = params_dict["policy"]
            params.batch_size = params_dict["batch_size"]
            params.learning_rate = params_dict["learning_rate"]
            params.buffer_size = params_dict["buffer_size"]
            params.gamma = params_dict["gamma"]
            params.tau = params_dict["tau"]
            params.train_freq = params_dict["train_freq"]
            params.gradient_steps = params_dict["gradient_steps"]
            params.learning_starts = params_dict["learning_starts"]
            return params
    except RuntimeError:
        return HyperParameters()
