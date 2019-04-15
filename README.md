# Carla-rl

## Installation

Download CARLA_0.8.2 from https://github.com/carla-simulator/carla/releases

Unzip CARLA_0.8.2.zip

Change directory to CARLA_0.8.2/PythonClient

Execute ```python setup.py  install```

## Dependencies

pip install gym

pip install tensorboardX

pip install atari_py

## Run Learning

### A2C Agent

python a2c_agent.py --env Carla-v0

### Async A2C Agent

python async_a2c_agent.py --env Carla-v0

### Learning Logs

tensorboard --logdir logs/

## Run Test

python a2c_agent.py --env Carla-v0 --test

## Render Simulator

To view execution set "render": True in ENV_CONFIG in carla_env.py

To run learning without rendering set "render": False in ENV_CONFIG in carla_env.py
