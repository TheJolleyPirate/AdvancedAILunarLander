from stable_baselines3.common.env_checker import check_env

from src.MemoryWrapper import MemoryWrapper
from src.novelty.NoveltyDirector import NoveltyDirector
from src.novelty.NoveltyName import NoveltyName

env = NoveltyDirector(NoveltyName.ORIGINAL).build_env(continuous=True)
env = MemoryWrapper(env)
print(check_env(env))