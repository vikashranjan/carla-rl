from gym.envs.registration import register
import envs
register(
    id='Carla-v0',
    entry_point='envs.carla_env:CarlaEnv',
)