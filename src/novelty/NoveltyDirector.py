from environments.atmosphereEnvironment import AtmosphereLunarLander
from environments.thrusterEnviro import FaultyThrusters
from src.environments.TurbulenceEnv import TurbulenceEnv
from src.environments.gravityEnvironment import GravityLunarLander
from src.novelty.NoveltyName import NoveltyName
from src.novelty.limited_sensor.LimitedSensor import LimitedSensor
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym


class NoveltyDirector:

    def __init__(self, novelty: NoveltyName = NoveltyName.ORIGINAL):
        self.novelty = novelty

    def build_env(self, render_mode=None, continuous: bool = True):
        if self.novelty == NoveltyName.FAULTY_THRUSTER:
            return Monitor(FaultyThrusters(render_mode=render_mode, continuous=continuous))
        if self.novelty == NoveltyName.ATMOSPHERE:
            return AtmosphereLunarLander(render_mode=render_mode, continuous=continuous)
        if self.novelty == NoveltyName.GRAVITY:
            return Monitor(GravityLunarLander(render_mode=render_mode, continuous=continuous))
        if self.novelty == NoveltyName.TURBULENCE:
            return Monitor(TurbulenceEnv(render_mode=render_mode, continuous=continuous))
        if self.novelty == NoveltyName.SENSOR:
            return LimitedSensor(render_mode=render_mode, continuous=continuous)
        else:
            return gym.make("LunarLander-v2", render_mode=render_mode, continuous=continuous)
