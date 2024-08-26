import math
from typing import Any

import gymnasium as gym
import pygame
from gymnasium import spaces
from gymnasium.core import ObsType, RenderFrame
import numpy as np
from pygame import gfxdraw

from src.novelty.ClonedLunaerLander import CloneLunarLander, SCALE


class LunarEnvironment(gym.Env):

    def __init__(self, render_mode="human", continuous: bool = True):
        self.env: CloneLunarLander = CloneLunarLander(render_mode=render_mode, continuous=continuous)
        self.render_mode = render_mode
        self.action_space = self.env.action_space
        self._adjust_observation_space()
        self.observe_distance = 0.3

    def _adjust_observation_space(self):
        assert self.env is not None
        low = self.env.observation_space.low.tolist()
        high = self.env.observation_space.high.tolist()
        # Add new observation
        #  bool: landing pad observed
        #  float: left landing pad position
        #  float: right landing pad position
        #  float: y of landing pad
        low.append(0)
        low.append(-1.5)
        low.append(-1.5)
        low.append(-1.5)
        high.append(1)
        high.append(1.5)
        high.append(1.5)
        high.append(1.5)
        self.observation_space = spaces.Box(np.array(low).astype(np.float32), np.array(high).astype(np.float32))

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        state = state.tolist()

        # get information from lunar environment
        helipad_x1 = self.env.helipad_x1
        helipad_x2 = self.env.helipad_x2
        helipad_y = self.env.helipad_y

        if not self.env.helipad_observed:
            pos_x, pos_y = self.env.lander.position
            diff_x = min(abs(helipad_x1 - pos_x), abs(helipad_x2 - pos_x))
            diff_y = abs(helipad_y - pos_y)
            # euclidian distance
            distance = math.sqrt(diff_x * diff_x + diff_y * diff_y)
            # update observed status of landing pad
            if distance <= self.observe_distance:
                state.append(1)
                self.env.helipad_observed = True
            else:
                state.append(0)
                state.append(helipad_x1)
                state.append(helipad_x2)
                state.append(helipad_y)
        if self.env.helipad_observed:
            state.append(1)
            state.append(helipad_x1)
            state.append(helipad_x2)
            state.append(helipad_y)
        return np.array(state, dtype=np.float32), reward, terminated, False, info


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        return self.env.reset()

    def close(self):
        self.env.close()


    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return
        if not self.helipad_observed:
            for x in [self.env.helipad_x1, self.env.helipad_x2]:
                x = x * SCALE
                flagy1 = self.env.helipad_y * SCALE
                flagy2 = flagy1 + 50
                pygame.draw.line(
                    self.env.surf,
                    color=(0, 0, 0),
                    start_pos=(x, flagy1),
                    end_pos=(x, flagy2),
                    width=1,
                )
                pygame.draw.polygon(
                    self.env.surf,
                    color=(0, 0, 0),
                    points=[
                        (x, flagy2),
                        (x, flagy2 - 10),
                        (x + 25, flagy2 - 5),
                    ],
                )
                gfxdraw.aapolygon(
                    self.env.surf,
                    [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                    (0, 0, 0),
                )
        return self.env.render()

