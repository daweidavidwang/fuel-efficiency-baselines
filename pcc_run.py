from env import Env

from pcc_ipopt.pcc import PCC_alg
from core.fuel_model_real import FuelModel as FMR
from core.fuel_model import FuelModel as FM
fuel_model_data_path = '/home/dawei/Downloads/0695e99d-a2ca-45d4-ace0-ab16e84cd94c.pkl'
env_config = {
    'ALTI_X': [0, 2000, 3000, 4000, 5000, 8000, 10000, 1e6],
    'ALTI': [0,    6,   9,   12,  9,  0,   -3,   0],
    'Mveh': 55e3,
    'target_v': 18.0,
    'ds': 100,
    'start_location': 0,
    'travel_distance': 10000,
    'obs_horizon':2000,
    'obs_step':20,
    'acc_constraints':[-0.1,0.1],
    'speed_constraints':[8.312872808159998, 21.68712719183999],
    'fuel_model':FMR(fuel_model_data_path)
}
pcc_config = {
    'ALTI_X': [0, 2000, 3000, 4000, 5000, 8000, 10000, 1e6],
    'ALTI': [0,    6,   9,   12,  9,  0,   -3,   0],
    'Mveh': 55e3,
    'target_v': 18.0,
    'ds': 100,
    'start_location': 0,
    'travel_distance': 10000,
    'obs_horizon':2000,
    'obs_step':20,
    'acc_constraints':[-0.1,0.1],
    'speed_constraints':[8.312872808159998, 21.68712719183999]
}
env = Env(env_config)
pcc = PCC_alg(pcc_config)
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