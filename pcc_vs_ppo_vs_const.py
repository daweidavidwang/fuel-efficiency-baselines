import ray 
from env import Env
from pcc_ipopt.pcc import PCC_alg
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from core.custom_logger import CustomLoggerCallback
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from core.fuel_model_real import FuelModel as FMR
from core.fuel_model import FuelModel as FM
ray.init(local_mode= True)

checkpoint_path = '/home/dawei/ray_results/PPO_train_rl_fuel_realdata/PPO_Env_34245_00000_0_2023-11-28_20-54-01/checkpoint_000050'
algo = Algorithm.from_checkpoint(checkpoint_path)
path = '/home/dawei/Downloads/content_1701161245375.pkl'
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
    'acc_constraints':[-0.15,0.15],
    'speed_constraints':[13.0, 27.0],
    'timeliness_check':True,
    'fuel_model':FMR(path)
}

pcc_config = {
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


obs, info = env.reset()
terminated = False
reward_list = []
obs_list = []
action_list = []

obs, info = env.reset()
terminated = False
truncated = False
while not (terminated or truncated):
    action = algo.compute_single_action(obs)
    action_list.extend([action])
    obs, reward, terminated, truncated, info = env.step(action)
    reward_list.extend([reward])


pcc = PCC_alg(pcc_config)
res = pcc.solve()

if res.success:
    v_list = list(res.x)
else:
    print("pcc cannot find optimal solution!")
    v_list = list(res.x)
print(reward_list)
print(action_list)
print(info['total_fuel'])

obs, info = env.reset()
terminated = False
reward_list_pcc = []

terminated = False
truncated = False
while not (terminated or truncated):
    curr_step = env.env_step
    action = (v_list[curr_step+1]**2 - v_list[curr_step]**2)/(2*env.ds)
    obs, reward, terminated, truncated, info = env.step(action)
    reward_list_pcc.extend([reward])

obs, info = env.reset()
terminated = False
reward_list_const = []
obs_list = []

obs, info = env.reset()
terminated = False
truncated = False
while not (terminated or truncated):
    action = 0.0
    obs, reward, terminated, truncated, info = env.step(action)
    reward_list_const.extend([reward])


import matplotlib.pyplot as plt

plt.plot(reward_list[:-1], label='ppo')
plt.plot(reward_list_const[:-1], label='const')
plt.plot(reward_list_pcc[:-1], label='pcc')
plt.legend()
plt.show()