from src.Novelty.NoveltyName import NoveltyName
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from src.Novelty.environments.TurbulenceEnv import TurbulenceEnv
from src.Novelty.environments.atmosphereEnvironment import AtmosphereLunarLander
from src.Novelty.environments.gravityEnvironment import GravityLunarLander
from src.Novelty.environments.limited_sensor.LimitedSensor import LimitedSensor
from src.Novelty.environments.thrusterEnviro import FaultyThrusters


class NoveltyDirector:

    def __init__(self, novelty: NoveltyName = NoveltyName.ORIGINAL):
        self.novelty = novelty

    def build_env(self, render_mode=None, continuous: bool = True):
        if self.novelty == NoveltyName.THRUSTER:
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
