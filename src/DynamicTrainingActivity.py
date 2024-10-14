from src.novelty.NoveltyName import NoveltyName
from src.training.DynamicTraining import Training


if __name__ == "__main__":
    try:
        dynamic_training = Training(
            novelty_name=NoveltyName.SENSOR,
            render=None,
            continuous=True,
            train_hour=4)
        dynamic_training.train()
    except KeyboardInterrupt:
        print("Training activity interrupted.")
        del dynamic_training
        print("Destructor finished.")


