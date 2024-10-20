import contextlib
import sys
import matplotlib.pyplot as plt

from src.evaluation.EvaluationResult import EvaluationResult
from src.evaluation.ModelEvaluation import evaluate
from src.exceptions.NoModelException import NoModelException
from src.novelty.NoveltyDirector import NoveltyDirector
from src.novelty.NoveltyName import TeamNovelty, OtherNovelty
from src.training.ModelAccess import ModelAccess


@contextlib.contextmanager
def suppress_print():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class MergeModel:
    def __init__(self, detecting_episodes: int):
        self.models = dict()
        for novelty in TeamNovelty:
            self.models[novelty] = ModelAccess(novelty)
        self.detecting_episodes = detecting_episodes

    def detect(self, evaluating_env):
        best_record = 1 - sys.maxsize
        best_model = None
        best_model_name = ""
        rewards = []
        for novelty in TeamNovelty:
            try:
                filename, model = self.models[novelty].load_best_model()
                evaluation_result: EvaluationResult = evaluate(model, evaluating_env, 20)
                # mean = evaluate(model, evaluating_env, self.detecting_episodes)
                rewards.append(evaluation_result.mean)
                if evaluation_result.mean > best_record:
                    best_record = evaluation_result.mean
                    best_model = model
                    best_model_name = filename
            except NoModelException as e:
                print(e.message)

        return best_model, best_model_name, rewards


def main():
    folder = "../results/"
    if not os.path.exists(folder):
        os.mkdir(folder)

    total_num_episodes = 100
    percentage_detect = 0.05
    detecting_episodes = int(total_num_episodes * percentage_detect / len(TeamNovelty))
    evaluating_episodes = total_num_episodes - detecting_episodes * len(TeamNovelty)
    merge_model = MergeModel(detecting_episodes)
    for novelty in list(TeamNovelty) + list(OtherNovelty):
        try:
            print(" ----------------- ")
            print(f"Evaluating environment {novelty.value.upper()} against all merged models")
            evaluating_env = NoveltyDirector(novelty).build_env(render_mode=None, continuous=True)
            # Detect which agent is best for solving the novelty
            model, model_name, rewards = merge_model.detect(evaluating_env)

            # Evaluate over the rest of the episodes
            with suppress_print():
                evaluation_result: EvaluationResult = evaluate(model, evaluating_env, evaluating_episodes, False)

            # Append detecting phase and evaluation phase rewards
            rewards += evaluation_result.rewards

            mean = sum(rewards) / total_num_episodes
            print(f"Achieved by {model_name}")
            print(f"Mean: {round(mean, 2)}")

            plt.plot(rewards, marker='o')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig(f"{folder}{novelty.value} rewards over {total_num_episodes} ep.png")
            plt.close()
        except (RuntimeError, ValueError):
            print(f"Evaluation skipped for {novelty.value} due to error(s).")
        except KeyboardInterrupt:
            print(f"Evaluation stopped at {novelty.value}.")


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
