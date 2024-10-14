from src.evaluation.ModelEvaluation import evaluate
from src.novelty.NoveltyDirector import NoveltyDirector
from src.novelty.NoveltyName import NoveltyName
from src.training.ModelAccess import ModelAccess


def main():
    env_novelty = NoveltyName.SENSOR
    model_novelty = NoveltyName.SENSOR

    # create env
    env = NoveltyDirector(env_novelty).build_env(render_mode="human")
    # load model
    model_access = ModelAccess(model_novelty, 0)
    _, model = model_access.load_best_model()

    evaluate(model, env, 10)


if __name__ == "__main__":
    main()
