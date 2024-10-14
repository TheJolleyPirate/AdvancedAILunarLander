from typing import Dict, Any


class HyperParameters:
    def __init__(
            self,
            n_timesteps: int = 50000,
            learning_rate: float = 0.001,
            buffer_size: int = 1_000_000,
            batch_size: int = 256,
            ent_coef: float = 0.1,
            gradient_steps: int = 1,
            gamma: float = 0.99,
            tau: float = 0.005,
            learning_starts: int = 100,
            use_sde: bool = False,
    ):
        self.policy_kwargs = {"net_arch": [400, 300]}
        self.policy = "MlpPolicy"
        self.n_timesteps = n_timesteps
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.tau = tau
        self.learning_starts = learning_starts
        self.use_sde = use_sde
        self.MAX_LEARNING_RATE = 0.01
        self.MIN_LEARNING_RATE = 1e-7

    def get_dict(self):
        return {"learning_rate": self.learning_rate, "batch_size": self.batch_size}


