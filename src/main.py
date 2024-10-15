import sys

from gymnasium.envs.box2d import LunarLander
from sympy.logic.boolalg import to_anf

from src.evaluation.ModelEvaluation import evaluate
from src.exceptions.NoModelException import NoModelException
from src.novelty.NoveltyDirector import NoveltyDirector
from src.novelty.NoveltyName import NoveltyName
from src.training.ModelAccess import ModelAccess


class MergeModel:
    def __init__(self, detecting_episodes: int):
        self.models = dict()
        for novelty in NoveltyName:
            self.models[novelty] = ModelAccess(novelty)
        self.detecting_episodes = detecting_episodes

    def detect(self, evaluating_env):
        best_record = 1 - sys.maxsize
        best_model = None
        best_model_name = ""
        total_rewards = 0
        for novelty in NoveltyName:
            try:
                filename, model = self.models[novelty].load_best_model()
                mean = evaluate(model, evaluating_env, self.detecting_episodes)
                total_rewards += mean * self.detecting_episodes
                if mean > best_record:
                    best_record = mean
                    best_model = model
                    best_model_name = filename
            except NoModelException as e:
                print(e.message)

        return best_model, best_model_name, total_rewards


def main():
    total_num_episodes = 100
    percentage_detect = 0.05
    detecting_episodes = int(total_num_episodes * percentage_detect / len(NoveltyName))
    evaluating_episodes = total_num_episodes - detecting_episodes * len(NoveltyName)
    merge_model = MergeModel(detecting_episodes)
    for novelty in NoveltyName:
        print(" ----------------- ")
        print(f"Evaluating environment {novelty.value.upper()} against all merged models")
        evaluating_env = NoveltyDirector(novelty).build_env(render_mode=None, continuous=True)
        model, model_name, detecting_rewards = merge_model.detect(evaluating_env)
        evaluated_mean = evaluate(model, evaluating_env, evaluating_episodes, False)
        total_rewards = evaluated_mean * evaluating_episodes + detecting_rewards
        mean = total_rewards / total_num_episodes
        print(f"Mean {round(mean, 2)} achieved by {model_name}")


if __name__ == "__main__":
    main()
