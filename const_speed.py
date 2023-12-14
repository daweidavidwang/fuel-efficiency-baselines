from env import Env
import random
import math
import numpy as np
from core.fuel_model_real import FuelModel as FMR
from core.fuel_model import FuelModel as FM

from core.slope_map import Slope

fuel_model_data_path = '/home/dawei/Documents/pkl_data/2d070af5-e7fd-4214-9968-69bd5a4643cb.pkl'
# mission_scenario = MissionLoader(fuel_model_data_path)
# mission_sceanrio_X, mission_sceanrio_height, mission_sceanrio_slope = mission_scenario.get_map_data()
# slope_map = Slope()
# # slope_map.construct(mode='slope', ALTI_X=mission_sceanrio_X, input_b=mission_sceanrio_slope, slope_range=[-0.0033845971195197774, 0.0033845971195197774])
# ALTI_X =  [0, 2000, 3000, 4000, 5000, 8000, 10000, 1e6]
# ALTI =  [0,    6,   9,   12,  9,  0,   -3,   0]
# slope_map.construct(mode='height', ALTI_X=ALTI_X, input_b=ALTI, slope_range=[-0.025183296730539186, 0.025183296730539186] )
fuel_model = FMR(fuel_model_data_path)
## real world map
ALTI_X, ALTI, ALTI_slope = fuel_model.get_map_data()
slope_map = Slope()
slope_map.construct(mode='slope', ALTI_X=ALTI_X, input_b=ALTI_slope, slope_range=[-0.005093949143751466, 0.005093949143751466] )


env_config = {
    'slope':slope_map,
    'Mveh': 55e3,
    'target_v': 21.0,
    'ds': 100,
    'start_location': 0,
    # 'start_location_range': [int(mission_sceanrio_X[0]),int(mission_sceanrio_X[-1])-11000],
    'travel_distance': 10000,
    'obs_horizon':2000,
    'obs_step':20,
    'acc_constraints':[-0.1,0.1],
    'speed_constraints':[8.312872808159998, 21.68712719183999],
    'fuel_model':fuel_model
}

env = Env(env_config)

obs, info = env.reset()
terminated = False
reward_list = []
obs_list = []
while True:
    obs, info = env.reset()
    print('start loc'+str(env.start_location))
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = 0.0
        obs, reward, terminated, truncated, info = env.step(action)
        reward_list.extend([reward])
    # print(info['total_fuel'])
    # env.monitor.plot()
    # print(info['const_baseline_fuel'])
    # print(reward_list)
