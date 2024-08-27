from src.novelty.NoveltyName import NoveltyName
from src.novelty.limited_sensor.LunarEnvironment import LunarEnvironment
import gymnasium as gym


class NoveltyDirector:

    def __init__(self, novelty: NoveltyName = NoveltyName.ORIGINAL):
        self.novelty = novelty

    def build_env(self):
        if self.novelty == NoveltyName.SENSOR:
            return LunarEnvironment()
        else:
            return gym.make("LunarLander-v2", render_mode="human", continuous=True)
