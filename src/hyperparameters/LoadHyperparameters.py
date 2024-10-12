from fileinput import filename
from typing import Optional

import yaml

from src.hyperparameters.Hyperparameters import HyperParameters


def load(file_name: Optional[str] = None):
    if file_name is None:
        return HyperParameters()
    try:
        with open(file_name) as file:
            # FIXME consider using package to load dictionary

            params_dict = yaml.safe_load(file)["LunarLanderContinuous-v2"]
            params = HyperParameters()
            # params.n_timesteps = params_dict["n_timesteps"]  # need to parse "!!float 5e5"
            params.policy = params_dict["policy"]
            params.batch_size = params_dict["batch_size"]
            params.learning_rate = params_dict["learning_rate"]
            params.buffer_size = params_dict["buffer_size"]
            params.gamma = params_dict["gamma"]
            params.tau = params_dict["tau"]
            params.gradient_steps = params_dict["gradient_steps"]
            params.learning_starts = params_dict["learning_starts"]
            return params
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return HyperParameters()



