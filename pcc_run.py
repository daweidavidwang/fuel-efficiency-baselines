from env import Env
import random
import math
import numpy as np
from pcc_ipopt.pcc import PCC_alg

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
    'speed_constraints':[1.0, 50.0]
}

env = Env(env_config)
pcc = PCC_alg(env_config)
res = pcc.solve()

if res.success:
    v_list = list(res.x)
else:
    print("pcc cannot find optimal solution!")
    v_list = list(res.x)
# v_list.extend([v_list[-1]])
obs, info = env.reset()
terminated = False
reward_list = []
obs_list = []
action_list = []

obs, info = env.reset()
terminated = False
truncated = False
while not (terminated or truncated):
    curr_step = env.env_step
    action = (v_list[curr_step+1]**2 - v_list[curr_step]**2)/(2*env.ds)
    action_list.extend([action])
    obs, reward, terminated, truncated, info = env.step(action)
    reward_list.extend([reward])

env.monitor.plot()
print(reward_list)
print(action_list)
print(info['total_fuel'])
