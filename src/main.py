import sys

from gymnasium.envs.box2d import LunarLander

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
        for novelty in NoveltyName:
            try:
                model = self.models[novelty].load_best_model()
                mean = evaluate(model, evaluating_env, self.detecting_episodes)
                if mean > best_record:
                    best_record = mean
                    best_model = model
            except NoModelException as e:
                print(e.message)

        return best_model


def main():
    total_num_episodes = 100
    percentage_detect = 0.05
    detecting_episodes = int(total_num_episodes * percentage_detect / len(NoveltyName))
    evaluating_episodes = total_num_episodes - detecting_episodes * len(NoveltyName)
    merge_model = MergeModel(detecting_episodes)
    for novelty in NoveltyName:
        print(f"Evaluating environment {novelty.value} against all merged models")
        evaluating_env = NoveltyDirector(novelty).build_env(render_mode=None, continuous=True)
        model = merge_model.detect(evaluating_env)
        evaluate(model, evaluating_env, evaluating_episodes, True)


if __name__ == "__main__":
    main()
