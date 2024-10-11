from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from src.hyperparameters.Hyperparameters import HyperParameters


class TrainingResult:

    def __init__(self, model: OffPolicyAlgorithm, params: HyperParameters, success: bool):
        self.model = model
        self.params = params
        self.success = success
