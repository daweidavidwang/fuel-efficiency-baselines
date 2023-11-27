from env import Env
import random
import math
import numpy as np

env_config = {
    'ALTI_X': [0, 2000, 3000, 4000, 5000, 8000, 10000, 1e6],
    'ALTI': [0,    20,   30,   40,  20,  -10,   0,   0],
    'Mveh': 55e3,
    'target_v': 72/3.6,
    'ds': 100,
    'start_location': 0,
    'travel_distance': 10000,
    'obs_horizon':2000,
    'obs_step':20,
    'acc_constraints':[-1,1],
    'speed_constraints':[1.0, 50.0],
    'timeliness_check':True
}

env = Env(env_config)

obs, info = env.reset()
terminated = False
reward_list = []
obs_list = []

obs, info = env.reset()
terminated = False
truncated = False
while not (terminated or truncated):
    action = (random.random()-0.5)*2
    obs, reward, terminated, truncated, info = env.step(action)
    reward_list.extend([reward])

env.monitor.plot()
print(sum(reward_list))
print(info['total_fuel'])
