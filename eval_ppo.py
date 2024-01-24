import ray 
from env import Env

from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from core.custom_logger import CustomLoggerCallback
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from core.fuel_model import FuelModel as FM
from core.slope_map import Slope
ray.init(local_mode= True)

fuel_model_data_path = '/home/dawei/Documents/pkl_data/2d070af5-e7fd-4214-9968-69bd5a4643cb.pkl'
fuel_model =FMR(fuel_model_data_path)

# slope_map = Slope()
# # slope_map.construct(mode='slope', ALTI_X=mission_sceanrio_X, input_b=mission_sceanrio_slope, slope_range=[-0.0033845971195197774, 0.0033845971195197774])
# ALTI_X =  [0, 2000, 3000, 4000, 5000, 8000, 10000, 1e6]
# ALTI =  [0,    -6,   -9,   -12,  -9,  0,   3,   0]
# slope_map.construct(mode='height', ALTI_X=ALTI_X, input_b=ALTI, slope_range=[-0.025183296730539186, 0.025183296730539186]   )

## real world map
# ALTI_X, ALTI, ALTI_slope = fuel_model.get_map_data()
# slope_map = Slope()
# slope_map.construct(mode='slope', ALTI_X=ALTI_X, input_b=ALTI_slope, slope_range=[-0.005093949143751466, 0.005093949143751466] )

ALTI_X, ALTI, ALTI_slope = fuel_model.get_map_data(factor=0,downsampling=1000)
slope_map = Slope()
# slope_map.construct(mode='slope', ALTI_X=ALTI_X, input_b=ALTI_slope, slope_range=[-0.005093949143751466, 0.005093949143751466] )
slope_map.construct(mode='height', ALTI_X=ALTI_X, input_b=ALTI, slope_range=[-0.005093949143751466, 0.005093949143751466] )


checkpoint_path = '/home/dawei/Documents/checkpoint_000200'
algo = Algorithm.from_checkpoint(checkpoint_path)

env_config = {
    'slope':slope_map,
    'Mveh': 55e3,
    'target_v': 21.0,
    'ds': 100,
    'start_location': 5000,
    'travel_distance': 100000,
    'obs_horizon':2000,
    'obs_step':20,
    'acc_constraints':[-0.05, 0.05] ,
    'speed_constraints':[18.922531942120045, 24.616176171879957],
    'timeliness_check':True,
    'fuel_model':fuel_model,
    'slope_range':fuel_model.slope_range
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
print(info['const_baseline_fuel'])