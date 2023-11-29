import ray 
from env import Env

from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from core.custom_logger import CustomLoggerCallback
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from core.fuel_model_real import FuelModel as FMR
from core.fuel_model import FuelModel as FM
ray.init(local_mode= True)

checkpoint_path = '/home/dawei/ray_results/PPO_train_rl_fuel_realdata/PPO_Env_8de39_00000_0_2023-11-29_18-32-11/checkpoint_000050'
algo = Algorithm.from_checkpoint(checkpoint_path)

path = '/home/dawei/Downloads/0695e99d-a2ca-45d4-ace0-ab16e84cd94c.pkl'
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
    'timeliness_check':True,
    'fuel_model':FMR(path)
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
    action = algo.compute_single_action(obs, explore=False)
    action_list.extend([action])
    obs, reward, terminated, truncated, info = env.step(action)
    reward_list.extend([reward])

env.monitor.plot()
print(reward_list)
print(action_list)
print(info['total_fuel'])
