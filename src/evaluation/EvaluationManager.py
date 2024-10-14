import sys
from typing import Dict

from gymnasium.envs.box2d import LunarLander
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from src.evaluation.ModelEvaluation import evaluate
from src.novelty.NoveltyName import NoveltyName


class EvaluationManager:

    def __init__(self, env: LunarLander, novelty_name: NoveltyName, n_episodes: int = 200):
        self._env = env
        self._novelty_name = novelty_name
        self._n_episodes = n_episodes
        self.models: Dict[str, OffPolicyAlgorithm] = dict()
        self.performance: Dict[str, float] = dict()
        self._latest_name: str = ""
        self._best_name: str = ""
        self._best_value: float = 1 - sys.maxsize
        print(f"EvaluationManager for {self._novelty_name.value} created.")

    def __del__(self):
        print(f"EvaluationManager for {self._novelty_name.value} is closed: \n "
              f" - Latest model: {self._latest_name}, (mean: {self.performance[self._latest_name]}) \n"
              f" - Best model: {self._best_name}, (mean: {self.performance[self._best_name]})")

        print("Detailed performance as following: ")
        for name, value in self.performance.items():
            print(f" - {name} : {value}")

    def has_model(self):
        return len(self.models) > 0

    def add_models(self, names, models):
        assert len(names) == len(models)
        print(f"Adding {len(names)} for evaluation: ", end="")
        for i in range(len(names)):
            print("#", end="")
            self.add_model(names[i], models[i])

        print(f"\nEvaluation complete.")

    def add_model(self, name, model) -> bool:
        if name in self.models.keys():
            return False
        self.models[name] = model
        self._evaluate(name, model)
        self._latest_name = name
        return True

    def _evaluate(self, name, model):
        value = evaluate(model, self._env, self._n_episodes)
        self.performance[name] = value

        # update best on the go
        if value > self._best_value:
            self._best_value = value
            self._best_name = name

    def get_best(self, re_calculate: bool = False):
        if len(self.models) == 0:
            return None

        if re_calculate:
            if len(self.models) == 0:
                return None
            current_value = 1 - sys.maxsize
            current_key = list(self.models.keys())[0]
            for key, model in self.models.items():
                if self.performance[key] > current_value:
                    current_value = self.performance[key]
                    current_key = key
            return current_key, self.models[current_key]
        else:
            return self._best_name, self.models[self._best_name]

    def get_latest(self):
        return self._latest_name, self.models[self._latest_name]

    def get_latest_name(self):
        return self._latest_name

    def get_best_name(self):
        return self._best_name


    def get_mean_reward(self, filename):
        return self.performance[filename]


