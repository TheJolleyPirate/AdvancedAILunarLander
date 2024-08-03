import yaml

from src.hyperparameters.Hyperparameters import HyperParameters


def load(filename: str):
    try:
        with open(filename) as file:
            params_dict = yaml.safe_load(file)
            params = HyperParameters()
            params.n_timesteps = params_dict["n_times"]
            params.learning_rate = params_dict["learning_rate"]
            params.buffer_size = params_dict["buffer_size"]
            params.batch_size = params_dict["batch_size"]
            params.ent_coef = params_dict["ent_coef"]
            params.train_freq = params_dict["train_freq"]
            params.gradient_steps = params_dict["gradient_steps"]
            params.gamma = params_dict["gamma"]
            params.tau = params_dict["tau"]
            params.learning_starts = params_dict["learning_starts"]
            params.use_sde = params_dict["use_sde"]
            params.policy_kwargs = params_dict["policy_kwargs"]
    except RuntimeError:
        return HyperParameters()
