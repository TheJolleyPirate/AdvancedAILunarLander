from typing import Dict, Any


class HyperParameters:
    def __init__(
            self,
            n_timesteps: float = 50000,
            policy: str = 'MlpPolicy',
            learning_rate: float = 0.01,
            buffer_size: int = 1_000_000,
            batch_size: int = 256,
            ent_coef: float = 0.1,
            train_freq: int = 1,
            gradient_steps: int = 1,
            gamma: float = 0.99,
            tau: float = 0.005,
            learning_starts: int = 100,
            use_sde: bool = False,
    ):
        self.policy_kwargs = {"net_arch": [400, 300]}
        self.n_timesteps = n_timesteps
        self.policy = policy
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.tau = tau
        self.learning_starts = learning_starts
        self.use_sde = use_sde
