# ============================================================================
# HYPERPARAMETERS & CONFIGURATION
# ============================================================================
# Adjust these values to control the learning algorithm and environment

# --- Environment Configuration ---
FAILURE_ANGLE_DEGREES = 90  # Pole angle limit before failure (default: 12°, larger = harder)
FORCE_MAGNITUDE = 75.0      # Maximum force that can be applied to cart (default CartPole: 10.0)
REWARD_TYPE = "swing_up"  # Reward function: "default", "angle_based", "comprehensive", "negative_angle" , "swing_up"

# Conditional override for swing_up
if REWARD_TYPE == "swing_up":
    FAILURE_ANGLE_DEGREES = 5000

# --- PID Controller Configuration ---
INTEGRAL_LIMIT = 20         # Maximum absolute value for integral term (prevents windup)
PID_TIMESTEP = 0.02         # Time step for PID calculations (matches environment tau=0.02)

# --- Genetic Algorithm - Population & Evolution ---
POPULATION_SIZE = 30        # Number of individuals per generation (larger = more exploration, slower)
ELITE_COUNT = 4             # Number of best individuals to keep each generation
GENERATIONS = 50            # Number of evolution cycles (more = longer search, better convergence)
EPISODES_PER_EVAL = 5       # Number of test runs to average for fitness (more = robust but slower)
MAX_STEPS_PER_EPISODE = 1000  # Maximum steps in each episode before truncation

# --- Genetic Algorithm - Parameter Search Ranges ---
# These define the search space for PID gains. Negative values allow corrective action.
KP_RANGE = (-500, 500)      # Proportional gain range: responds to current error
KI_RANGE = (-500, 500)      # Integral gain range: responds to accumulated error over time
KD_RANGE = (-500, 500)      # Derivative gain range: responds to rate of change of error

# --- Genetic Algorithm - Evolution Parameters ---
MUTATION_RATE = 0.9        # Probability of mutating each gene (0.0-1.0, lower = more stable)
MUTATION_EFFECT = 0.5       # Size of mutations as fraction of current value (lower = finer tuning)
MUTATION_EFFECT_DECAY = 0.96  # Decay factor for mutation effect over generations
CROSSOVER_RATE = .95        # Probability of crossover between parents (0.0-1.0)
ELITISM = True              # Keep best individual in next generation (recommended: True)

# --- Visualization Settings ---
VISUALIZE_ALL_INDIVIDUALS_DURING_TRAINING = False  # Show all individuals or just best per generation
VISUALIZE_BEST = True  # If True, dont visualize the best individual of each generation 

RENDER_EVERY_N_STEPS = 10    # Update visualization every N steps (lower = smoother but slower)
