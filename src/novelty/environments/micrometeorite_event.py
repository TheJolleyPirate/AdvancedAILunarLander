import gymnasium as gym
import numpy as np

class MicrometeoriteEventWrapper(gym.Wrapper):
    """
    A custom environment wrapper that simulates micrometeorite impact events in the LunarLander environment.
    Micrometeorites fall towards the lunar surface with defined speed and trajectory, and if they collide with the lander,
    they can alter its position, angle, and engine effects. The novelty introduces randomness to the lander's landing process,
    increasing the challenge of navigating and controlling the rover.

    Parameters:
    env: gym.Env - the original LunarLander environment.
    meteorite_params: dict - optional parameters for controlling the meteorite speed and movement.
    """
    def __init__(self, env, meteorite_params=None):
        super().__init__(env)

        # Set default meteorite parameters (if none are provided)
        if meteorite_params is None:
            meteorite_params = {'speed': 0.2}  # Speed of meteorite falling

        # Initial position, falling speed, and horizontal movement direction of meteorites
        self.initial_positions = [
            {"x": -1.3, "y": 2.2, "speed": meteorite_params['speed'], "dir_x": 0.33},
            {"x": -1.2, "y": 1.8, "speed": meteorite_params['speed'], "dir_x": 0.44},
            {"x": -1.2, "y": 1.5, "speed": meteorite_params['speed'], "dir_x": 0.22},
            {"x": -0.9, "y": 1.5, "speed": meteorite_params['speed'], "dir_x": 0.33},
            {"x": -0.8, "y": 1.5, "speed": meteorite_params['speed'], "dir_x": 0.11}
        ]

        # Copy the initial positions to current meteorite positions
        self.meteorite_positions = [pos.copy() for pos in self.initial_positions]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update meteorite positions
        for i, meteorite in enumerate(self.meteorite_positions):
            meteorite['y'] -= meteorite['speed']  # Meteorite falls
            meteorite['x'] += meteorite['dir_x']  # Meteorite moves along the x-axis

            # Reset position if the meteorite falls to the ground or goes out of the x-axis bounds
            if meteorite['y'] <= 0 or meteorite['x'] < -1.5 or meteorite['x'] > 1.5:
                meteorite['y'] = self.initial_positions[i]['y']  # Reset y position
                meteorite['x'] = self.initial_positions[i]['x']  # Reset x position

        # Apply meteorite impact on collision with the lander
        obs, action = self.apply_meteorite_impact_on_collision(obs, action)

        return obs, reward, terminated, truncated, info

    def detect_collision(self, obs):
        lander_x = obs[0]  # Get the lander's x-coordinate
        lander_y = obs[1]  # Get the lander's y-coordinate

        # Check each meteorite to see if it is close to the lander
        for meteorite in self.meteorite_positions:
            collision_x = abs(lander_x - meteorite['x']) < 0.15  # Check proximity on x-axis
            collision_y = abs(lander_y - meteorite['y']) < 0.15   # Check proximity on y-axis
            if collision_x and collision_y:
                return True

        return False

    def apply_meteorite_impact_on_collision(self, obs, action):
        # 检测碰撞，并且只在碰撞时对月球车施加影响
        if self.detect_collision(obs):
            for meteorite in self.meteorite_positions:
                impact_strength = meteorite['speed']
                obs[0] += 1 * impact_strength  # Impact on x position
                obs[4] += -0.4 * impact_strength  # Impact on angle (rotational effect)

                # Impact on engine power
                if action == 2:  # Main engine
                    engine_impact_factor = 2.5
                    action = np.clip(action * (1 - engine_impact_factor * impact_strength), 0, 1)

        return obs, action

    def reset(self, **kwargs):
        # Resets the environment and all meteorite positions to their initial state.
        self.meteorite_positions = [pos.copy() for pos in self.initial_positions]
        return self.env.reset(**kwargs)