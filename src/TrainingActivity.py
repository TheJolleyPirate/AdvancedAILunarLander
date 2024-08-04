import gymnasium as gym
from src.Novelty.NoveltyName import NoveltyName
from src.exceptions.NoModelException import NoModelException
from src.training.ModelTraining import continueTrainingModel, trainNewModel


def training_activity(novel_name=NoveltyName.ORIGINAL):
    env = gym.make("LunarLander-v2", render_mode="human", continuous=True)
    for _ in range(10):
        try:
            model = continueTrainingModel(NoveltyName.ORIGINAL)
        except NoModelException:
            model = trainNewModel(env, NoveltyName.ORIGINAL)


if __name__ == "__main__":
    training_activity()
