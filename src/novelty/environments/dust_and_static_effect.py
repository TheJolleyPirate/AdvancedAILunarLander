import gymnasium as gym
import numpy as np


# Define the Dust And staticEffectWrapper class, which inherits from gym.Wrapper, to encapsulate the environment and
# add dust and static effects.
class DustAndstaticEffectWrapper(gym.Wrapper):
    """
    A custom environment wrapper designed to simulate the effects of lunar dust and static effect on a lander.
    Parameters:
    env: gym.Env
        The base environment to be wrapped. This environment will have the dust and static effects applied to it.

    dust_threshold: float
        The height threshold at which dust effects begin to impact the lander's attitude.
        This value typically ranges from 0 to 1, representing a proportion of the total height.
        For example, 0.5 indicates that the dust effects start when the height falls below 50% of the total height.

    max_dust_effect: float
        The maximum dust disturbance effect. This value represents the maximum amount of random disturbance that the
        dust can cause to the lander's attitude.
        Higher values mean a stronger disturbance effect. This value typically ranges from 0 to 1, indicating the
        proportion of the disturbance intensity.
        For example, 0.5 indicates that the maximum disturbance can reach 50% of the attitude value.

    static_probability: float
        The probability that static effects will cause the engine to briefly fail.This value represents the likelihood
        of engine failure at each step.
        It ranges from 0 to 1, with 0 meaning no effect and 1 meaning the engine fails at every step.
        For example, 0.3 indicates a 30% probability of engine failure at each time step.
    """
    def __init__(self, env, dust_threshold=0.5, max_dust_effect=0.5, static_probability=0.3):
        # Call the constructor of the parent class gym.Wrapper to initialize the environment
        super().__init__(env)
        self.dust_threshold = dust_threshold  # Sets the height threshold that triggers the dust effect
        self.max_dust_effect = max_dust_effect  # Sets the maximum dust effect (impact on angular velocity)
        # Sets the maximum probability that static will affect the engine
        self.static_probability = static_probability
        self.current_dust_effect = 0  # Initializes the current dust effect strength
        self.current_static_probability = 0  # Initializes the current static probability

    # Define the step method, extending the original environment's step method to add the effects of dust and static.
    def step(self, action):
        # Call the step method of the original environment to obtain the current state observation, reward,
        # whether it is terminated, whether it is truncated, and other information info.
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Extract the current altitude of the lander (usually the second element of observation)
        height = observation[1]
        # If the current altitude is below the dust threshold, the dust effect is triggered.
        if height < self.dust_threshold:
            # Dynamically increase the strength of the dust effect
            self.current_dust_effect = min(
                self.current_dust_effect + 0.01, self.max_dust_effect
            )
            # Calculate the strength of the dust's effect on angular velocity, which increases with decreasing altitude
            dust_effect = (self.dust_threshold - height) / self.dust_threshold * self.current_dust_effect
            # Adds randomly generated dust effects to angular velocity of lander.
            observation[2] += np.random.uniform(-dust_effect, dust_effect)

            # Dynamically increase the probability of static effect
            self.current_static_probability = min(
                self.current_static_probability + 0.01, self.static_probability
            )
            # Triggers the static effect with a certain probability.
            if np.random.uniform(0, 1) < self.current_static_probability:
                # The engine number to be disabled is randomly selected (1: left engine, 2: main engine, 3: right
                # engine), and the number of engines disabled is also random (1 to 3 engines).
                engines_to_disable = np.random.choice([1, 2, 3], size=np.random.randint(1, 4), replace=False)
                # If the current action matches a disabled engine, set the action to 0 (i.e., do not perform any
                # engine operation).
                if action in engines_to_disable:
                    action = 0
        # Call the step method of the original environment again and return the processed result
        return self.env.step(action)

    # Override the reset method and call the original environment's reset method directly to ensure the initial state
    # of the environment at each reset.
    def reset(self, **kwargs):
        # Reset dust and static effects when the environment resets
        self.current_dust_effect = 0  # Reset the current dust effect strength
        self.current_static_probability = 0  # Reset the current static probability
        return self.env.reset(**kwargs)
