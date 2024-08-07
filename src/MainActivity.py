from src.Novelty.NoveltyName import NoveltyName
from src.training.ModelAccess import loadModel

# Set up environment for lunar lander
# env = gym.make("LunarLander-v2", render_mode="human", continuous=True)

# Load latest trained model
model = loadModel(NoveltyName.ORIGINAL)
env = model.get_env()
observation = env.reset()

for _ in range(10000):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)

    if done:
        env = model.get_env()
        observation = env.reset()

env.close()
