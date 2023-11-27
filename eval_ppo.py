import ray 
from env import Env

from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from core.custom_logger import CustomLoggerCallback
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm

ray.init(local_mode= True)

checkpoint_path = '/home/david/ray_results/PPO_train_rl_fuel/PPO_Env_6c277_00000_0_2023-11-27_17-00-33/checkpoint_000050'
algo = Algorithm.from_checkpoint(checkpoint_path)

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
action_list = []

obs, info = env.reset()
terminated = False
truncated = False
while not (terminated or truncated):
    action = algo.compute_single_action(obs)
    action_list.extend([action])
    obs, reward, terminated, truncated, info = env.step(action)
    reward_list.extend([reward])

env.monitor.plot()
print(reward_list)
print(action_list)
print(info['total_fuel'])
