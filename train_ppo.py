import ray 
from env import Env
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from core.custom_logger import CustomLoggerCallback
from ray import tune
from core.fuel_model_real import FuelModel
ray.init(num_gpus=1, num_cpus=16)
# ray.init(local_mode=True)
config = PPOConfig()
# Update the config object.
# config = config.training(lr=tune.grid_search([0.001, 0.0001]))
config = config.training(lr=1e-4, train_batch_size = 2048*64*2, sgd_minibatch_size=2048*32*2)
# lr_schedule=[[0, 1e-4], [20000000, 1e-5]]
# config = config.training(lr=tune.grid_search([ 1e-6,  5e-6, 1e-3, 1e-4, 1e-5]), train_batch_size = 2048*64*2, sgd_minibatch_size=2048*32*2)

fuel_model_data_path = '/home/dawei/Downloads/0695e99d-a2ca-45d4-ace0-ab16e84cd94c.pkl'
# Set the config object's env.
config = config.environment(env=Env,env_config={
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
    'fuel_model':FuelModel(fuel_model_data_path)
}
)
config = config.rollouts(num_rollout_workers=15, rollout_fragment_length="auto")
config = config.callbacks(CustomLoggerCallback)
config = config.resources(num_gpus=1, num_cpus_per_worker=1)
# Use to_dict() to get the old-style python config dict
# when running with tune.
tune.Tuner(  
    "PPO",
    run_config=air.RunConfig(name='PPO_train_rl_fuel_realdata', \
    stop={"training_iteration": 100}, verbose=3, log_to_file=True, 
        checkpoint_config=air.CheckpointConfig(
            num_to_keep = 40,
            checkpoint_frequency = 5
        ),
        ),
    param_space=config.to_dict(),
).fit()
