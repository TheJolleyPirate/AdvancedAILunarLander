from src.novelty.NoveltyName import NoveltyName, TeamNovelty, OtherNovelty
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import gymnasium as gym
from src.novelty.environments.TurbulenceEnv import TurbulenceEnv
from src.novelty.environments.atmosphereEnvironment import AtmosphereLunarLander
from src.novelty.environments.gravityEnvironment import GravityLunarLander
from src.novelty.environments.limited_sensor.LimitedSensor import LimitedSensor
from src.novelty.environments.thrusterEnviro import FaultyThrusters

from src.novelty.environments.lunar_lander_asteroid import LunarLanderAsteroidNovelty
from src.novelty.environments.lunar_lander_turret import LunarLanderTurret
from src.novelty.environments.lunar_lander_overhang import LunarLanderOverhang
from src.novelty.environments.lunar_lander_blackhole import LunarLanderForce
from src.novelty.environments.lunar_lander_windy_chasms import LunarLanderWindyChasms

from src.novelty.environments.delay_env import ActionDelayWrapper
from src.novelty.environments.dust_and_static_effect import DustAndstaticEffectWrapper
from src.novelty.environments.micrometeorite_event import MicrometeoriteEventWrapper
from src.novelty.environments.reduced_visibility_env import DustVisibilityWrapper


class NoveltyDirector:

    def __init__(self, novelty: NoveltyName = TeamNovelty.ORIGINAL):
        self.novelty = novelty

    def build_env(self, render_mode=None, continuous: bool = True):
        env = self._find_env(render_mode=render_mode, continuous=continuous)
        env.num_envs = 1
        # return Monitor(env)
        return env

    def _find_env(self, render_mode=None, continuous: bool = True):

        if self.novelty == TeamNovelty.THRUSTER:
            return FaultyThrusters(render_mode=render_mode, continuous=continuous)
        if self.novelty == TeamNovelty.ATMOSPHERE:
            return AtmosphereLunarLander(render_mode=render_mode, continuous=continuous)
        if self.novelty == TeamNovelty.GRAVITY:
            return GravityLunarLander(render_mode=render_mode, continuous=continuous)
        if self.novelty == TeamNovelty.TURBULENCE:
            return TurbulenceEnv(render_mode=render_mode, continuous=continuous)
        if self.novelty == TeamNovelty.SENSOR:
            return LimitedSensor(render_mode=render_mode, continuous=continuous)
        # Novelties from other teams
        if self.novelty == OtherNovelty.ASTEROID:
            return LunarLanderAsteroidNovelty(render_mode=render_mode, continuous=continuous)
        if self.novelty == OtherNovelty.BLACKHOLE:
            return LunarLanderForce(render_mode=render_mode, continuous=continuous)
        if self.novelty == OtherNovelty.WIND:
            return LunarLanderWindyChasms(render_mode=render_mode, continuous=continuous)
        if self.novelty == OtherNovelty.TURRET:
            return LunarLanderTurret(render_mode=render_mode, continuous=continuous)
        if self.novelty == OtherNovelty.OVERHANG:
            return LunarLanderOverhang(render_mode=render_mode, continuous=continuous)

        # environment wrappers
        env = gym.make("LunarLander-v2", render_mode=render_mode, continuous=continuous)
        env = Monitor(env)
        if self.novelty == OtherNovelty.DELAY:
            return ActionDelayWrapper(env)
        if self.novelty == OtherNovelty.DUST_STATIC:
            return DustAndstaticEffectWrapper(env)
        if self.novelty == OtherNovelty.MICROMETEORITE:
            return MicrometeoriteEventWrapper(env)
        if self.novelty == OtherNovelty.REDUCED_VISIBILITY:
            return DustVisibilityWrapper(env)

        # original environment
        else:
            return gym.make("LunarLander-v2", render_mode=render_mode, continuous=continuous)
