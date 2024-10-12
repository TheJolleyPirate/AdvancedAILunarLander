from src.novelty.NoveltyName import NoveltyName
from src.training.Training import Training


if __name__ == "__main__":
    try:
        activity = Training(
            novelty_name=NoveltyName.SENSOR,
            render=None,
            continuous=True,
            train_hour=4)
        activity.train()
    except KeyboardInterrupt:
        print("Training activity interrupted.")
        del activity
        print("Destructor finished.")
    finally:
        del activity
