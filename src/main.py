import sys

from gymnasium.envs.box2d import LunarLander
from sympy.logic.boolalg import to_anf

from src.evaluation.EvaluationResult import EvaluationResult
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
                evaluation_result: EvaluationResult = evaluate(model, evaluating_env, 20)
                # mean = evaluate(model, evaluating_env, self.detecting_episodes)
                total_rewards += evaluation_result.mean * self.detecting_episodes
                if evaluation_result.mean > best_record:
                    best_record = evaluation_result.mean
                    best_model = model
                    best_model_name = filename
            except NoModelException as e:
                print(e.message)

        return best_model, best_model_name, total_rewards


def main():
    total_num_episodes = 20
    percentage_detect = 0.05
    detecting_episodes = int(total_num_episodes * percentage_detect / len(NoveltyName))
    evaluating_episodes = total_num_episodes - detecting_episodes * len(NoveltyName)
    merge_model = MergeModel(detecting_episodes)
    for novelty in NoveltyName:
        print(" ----------------- ")
        print(f"Evaluating environment {novelty.value.upper()} against all merged models")
        evaluating_env = NoveltyDirector(novelty).build_env(render_mode=None, continuous=True)
        model, model_name, detecting_rewards = merge_model.detect(evaluating_env)
        evaluated_mean, std_reward, min_reward, max_reward = evaluate(model, evaluating_env, evaluating_episodes, False)
        total_rewards = evaluated_mean * evaluating_episodes + detecting_rewards
        mean = total_rewards / total_num_episodes
        print(f"Achieved by {model_name}")
        print(f"Mean: {round(mean, 2)}")
        print(f"Standard deviation: {std_reward}")
        print(f"Min: {min_reward}")
        print(f"Max: {max_reward}")



if __name__ == "__main__":
    main()
