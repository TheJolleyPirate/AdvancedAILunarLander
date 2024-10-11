from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from src.hyperparameters.Hyperparameters import HyperParameters


def get_hyperparameters(model: OffPolicyAlgorithm):
    default_params = HyperParameters()
    return HyperParameters(
        n_timesteps=default_params.n_timesteps if model.num_timesteps is None else model.num_timesteps,
        policy=default_params.policy if model.policy is None else model.policy,
        learning_rate=default_params.learning_rate if model.learning_rate is None else model.learning_rate,
        buffer_size=default_params.buffer_size if model.buffer_size is None else model.buffer_size,
        batch_size=default_params.batch_size if model.batch_size is None else model.batch_size,
        train_freq=default_params.train_freq if model.train_freq is None else model.train_freq,
        gradient_steps=default_params.gradient_steps if model.gradient_steps is None else model.gradient_steps,
        gamma=default_params.gamma if model.gamma is None else model.gamma,
        tau=default_params.tau if model.tau is None else model.tau,
        learning_starts=default_params.learning_starts if model.learning_starts is None else model.learning_starts,
        use_sde=default_params.use_sde if model.use_sde is None else model.use_sde)


def set_hyperparameters(params: HyperParameters, model: OffPolicyAlgorithm):
    model.num_timesteps = params.n_timesteps
    model.policy = params.policy
    model.learning_rate = params.learning_rate
    model.buffer_size = params.buffer_size
    model.batch_size = params.batch_size
    model.train_freq = params.train_freq
    model.gradient_steps = params.gradient_steps
    model.gamma = params.gamma
    model.tau = params.tau
    model.learning_starts = params.learning_starts
    model.use_sde = params.use_sde


def scale_learning_rate(params: HyperParameters, scaler: float):
    new_rate = params.learning_rate * scaler
    new_rate = min(params.MAX_LEARNING_RATE, new_rate)
    params.learning_rate = new_rate


def scale_batch_size(params: HyperParameters, scaler: float):
    new_size: int = int(params.batch_size * scaler)
    new_size = max(1, new_size)
    params.batch_size = new_size
