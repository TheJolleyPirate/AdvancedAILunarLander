
from datetime import datetime, timedelta
from time import sleep

from src.hyperparameters.Hyperparameters import HyperParameters
from src.novelty.NoveltyName import NoveltyName
from src.novelty.NoveltyDirector import NoveltyDirector
from src.exceptions.NoModelException import NoModelException
from src.training.ModelAccess import ModelAccess
from src.training.ModelTraining import trainNewModel, train_model
from src.training.TuningParameters import scale_learning_rate, scale_batch_size


class Training:
    def __init__(self,
                 novelty_name: NoveltyName = NoveltyName.ORIGINAL,
                 render=None,
                 continuous=True,
                 train_hour=4):
        self.novelty_name = novelty_name
        self.render = render
        self.continuous = continuous
        self.train_hour = train_hour
        self.model_access = ModelAccess(novelty_name)

    def __del__(self):
        try:
            del self.model_access
        except AttributeError:
            print("model_access object not created.")

    def train(self):
        # setting up environment
        env = NoveltyDirector(self.novelty_name).build_env(self.render, self.continuous)

        # dynamic updatable params
        params: HyperParameters = HyperParameters()
        recent_success = []
        recent_files = []
        queue_size = 10
        prev_success = True
        exploit = True

        start_time = datetime.now()
        end_time = start_time + timedelta(hours=self.train_hour)
        while datetime.now() < end_time:
            try:
                # Load model
                if not self.model_access.has_model():
                    raise NoModelException(novelty_name=self.novelty_name)

                # Re-train model
                if prev_success or not exploit:
                    name, model = self.model_access.load_latest_model(params)
                    print(f"Latest model {name} is loaded.")
                else:
                    name, model = self.model_access.load_best_model(params)
                    print(f"Best model {name} is loaded.")
                result = train_model(env=env,
                                     prev_model=model,
                                     prev_filename=name,
                                     prev_mean=self.model_access.get_mean_reward(name),
                                     model_novelty=self.novelty_name)

                self.model_access.add_model(result.filename, result.model)

                # Update records
                params = result.params
                prev_success = result.success
                recent_success.append(result.success)
                recent_files.append(result.filename)
                if len(recent_success) > queue_size:
                    recent_success.pop(0)
                    recent_files.pop(0)

                # Adjust hyper parameters
                count = [recent_files.count(s) for s in set(recent_files)]

                # if the same file is training all the time
                if len(recent_success) == queue_size and len(count) == 1:
                    exploit = False

                latest_name = self.model_access.get_latest_name()
                best_name = self.model_access.get_best_name()
                if self.model_access.get_performance(best_name) - self.model_access.get_performance(latest_name) <= 20:
                    prev_success = True

                if exploit:
                    # exploit
                    scale_learning_rate(params, 0.95)
                    scale_batch_size(params, 2)
                else:
                    # explore
                    scale_learning_rate(params, 1.5)
                    scale_batch_size(params, 0.25)

                sleep(20)

            except NoModelException:
                result = trainNewModel(env, self.novelty_name, params)
                self.model_access.add_model(result.filename, result.model)

