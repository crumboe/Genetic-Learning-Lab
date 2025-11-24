
import gymnasium as gym
import numpy as np

# Create a wrapper to customize the reward function
class CustomRewardCartPole(gym.Wrapper):
    def __init__(self, env, reward_type="angle_based"):
        """
        reward_type options:
        - "default": Standard +1 per step
        - "angle_based": Reward based on how upright the pole is
        - "comprehensive": Considers angle, angular velocity, and cart position
        - "negative_angle": Penalize based on angle deviation
        - "swing_up": Start pole hanging down, swing up and balance
        """
        super().__init__(env)
        self.reward_type = reward_type

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        if self.reward_type == "swing_up":
            # Modify the state to start hanging down
            # state format: [x, x_dot, theta, theta_dot]
            state = np.array(state, dtype=np.float32)
            state[2] = np.pi  # Start hanging down (180 degrees)
            # Reset the environment's internal state
            self.env.unwrapped.state = tuple(state)
        return state, info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        # Extract state variables
        x, x_dot, theta, theta_dot = state

        # Customize reward based on type
        if self.reward_type == "default":
            custom_reward = reward

        elif self.reward_type == "angle_based":
            # Reward is higher when pole is more upright (theta closer to 0)
            # Max reward of 1 when perfectly upright, decreases with angle
# DEFINE YOUR CUSTOM REWARD LOGIC HERE
            # custom_reward = 
            NotImplementedError("Angle-based reward function not implemented.") # remove this line when implementing

        elif self.reward_type == "comprehensive":
            # Combination of multiple factors (balanced for learning)
# DEFINE YOUR CUSTOM REWARD LOGIC HERE
            # angle_reward =       # Upright bonus
            # velocity_penalty =   # Small penalty for angular velocity
            # position_penalty =   # Small penalty for position
            
            # Keep rewards positive and scaled appropriately
            # custom_reward = max(0, angle_reward + velocity_penalty + position_penalty)
            NotImplementedError("Angle-based reward function not implemented.") # remove this line when implementing
        elif self.reward_type == "negative_angle":
            # Penalize angle deviation (negative reward when tilted)
            custom_reward = -abs(theta)
        elif self.reward_type == "custom":
            # User-defined custom reward function
            
            NotImplementedError("Custom reward function not implemented.") # remove this line when implementing
                #=============================================================================================================
                #
                # You can define your own custom reward function logic here. Change the reward type to "custom" in the
                # hyperparameters section to use it.
                #=============================================================================================================
        elif self.reward_type == "swing_up":
            # Two-phase control: energy injection when far from vertical, balance control when near vertical
            
            # Normalize theta to [-pi, pi] range for consistent calculations
            theta_normalized = np.arctan2(np.sin(theta), np.cos(theta))
            
            # Calculate system energy (simplified inverted pendulum energy)
            # Kinetic energy: (1/2) * m * l^2 * theta_dot^2
            # Potential energy: m * g * l * (1 - cos(theta))
            # For upright position: E_target ≈ 2 * m * g * l
            length = self.env.unwrapped.length
            masspole = self.env.unwrapped.masspole
            gravity = self.env.unwrapped.gravity
            
            kinetic_energy = 0.5 * masspole * (length ** 2) * (theta_dot ** 2)
            potential_energy = masspole * gravity * length * (np.cos(theta))
            total_energy = kinetic_energy + potential_energy
            target_energy = 2 * masspole * gravity * length  # Energy at upright position
            
            # Phase detection: far from vertical vs near vertical
            angle_from_upright = abs(theta_normalized)  # Distance from upright (0)
            

            if angle_from_upright > np.pi / 4:  # More than 30 degrees from vertical - SWING UP PHASE
                # Energy-based reward: want to pump energy into the system
                energy_error = (-total_energy + target_energy)
                pumping_direction =- np.sign(theta_dot * np.cos(theta))
                state[2] = energy_error*pumping_direction
                # if energy_error > 0: # if there isnt enough energy reward it towards to right number
                energy_reward = 1.0 - min(abs(energy_error) / target_energy, 1.0)
                # else: # if too much energy punish
                #     energy_reward = -1-min(abs(energy_error) / target_energy, 1.0) 

                # Small bonus for moving toward vertical

                upright_progress = .5 - (angle_from_upright / np.pi)
                
                # Penalize cart moving too far from center
                position_penalty = -0.1 * (x ** 2)
                
                custom_reward = 2.0 * energy_reward + 5 * upright_progress + position_penalty
                
            else:  # Within 30 degrees of vertical - BALANCE PHASE
                state[2] = theta_normalized*2 + x*0.3
                # Switch to balance control: minimize angle and angular velocity
                angle_reward = 1.0 - (angle_from_upright / (np.pi / 4))  # Reward being close to vertical
                stability_reward = 1.0 - min(abs(theta_dot), 1.0)  # Penalize angular velocity
                energy_error = abs(total_energy - target_energy)
                energy_reward = 1.0 - min(energy_error / target_energy, 1.0)
                # Keep cart centered
                position_reward = 1.0 - min(abs(x) / 2.4, 1.0)
                cart_velocity_penalty = - (x_dot ** 2)

                # Heavy weighting on balance once we're close
                custom_reward = 5.0 * angle_reward + 2.0 * stability_reward + position_reward + cart_velocity_penalty
                custom_reward *= 2.0  # Scale up balance rewards

        elif self.reward_type == "custom_15_deg":
            # Custom reward function that gives a bonus for keeping the pole within 15 degrees
            # and keeping the cart centered
            
            # Target: 15 degrees (convert to radians)
            target_angle = np.deg2rad(15)
            angle_degrees = abs(np.rad2deg(theta))
            
            # Angle reward: exponential falloff from target angle
            # Peak reward when at exactly 15 degrees
            angle_error = abs(abs(theta) - target_angle)
            angle_reward = np.exp(-10 * angle_error**2)  # Sharp peak at 15 degrees
            
            # Stability reward: penalize angular velocity (want to hold steady)
            stability_reward = np.exp(-5 * theta_dot**2)
            
            # Position reward: keep cart centered
            # Exponential falloff from center position
            position_reward = np.exp(-2 * x**2)
            
            # Cart velocity penalty: discourage excessive movement
            velocity_penalty = np.exp(-0.5 * x_dot**2)
            
            # Combine all components with weights
            # Prioritize angle (most important), then position, then stability
            custom_reward = (
                2.0 * angle_reward +       # Weight: 2.0 - most important
                1.0 * position_reward +     # Weight: 1.0 - keep centered
                0.5 * stability_reward +    # Weight: 0.5 - reduce oscillation
                0.3 * velocity_penalty      # Weight: 0.3 - smooth movement
            )
            # dist_error_normalized = x / self.env.unwrapped.x_threshold
            # angle_error_normalized = angle_error / np.deg2rad(FAILURE_ANGLE_DEGREES)
            state[2] = target_angle-theta # for the PID controller
            # Normalize to reasonable range (0 to ~4)
            
            # This helps the GA distinguish between good and bad solutions
            
        else:
            custom_reward = reward

        return state, custom_reward, terminated, truncated, info

