from config import *
import numpy as np 

# PID Controller class
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.integral = 0
        self.prev_error = 0
        self.integral_limit = INTEGRAL_LIMIT  # Use hyperparameter

    def reset(self):
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt=None):
        if dt is None:
            dt = PID_TIMESTEP  # Use hyperparameter

        self.integral += error * dt
        # Anti-windup: clamp integral term
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        return output
