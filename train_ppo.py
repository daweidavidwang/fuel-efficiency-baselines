import ray 
from env import Env

from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from core.custom_logger import CustomLoggerCallback
from ray import tune
ray.init(num_gpus=1, num_cpus=16)
# ray.init(local_mode=True)
config = PPOConfig()
# Update the config object.
# config = config.training(lr=tune.grid_search([0.001, 0.0001]))
config = config.training(lr=1e-4, train_batch_size = 2048*64*4, sgd_minibatch_size=2048*32*4)
# Set the config object's env.
config = config.environment(env=Env,env_config={
    'ALTI_X': [0, 2000, 3000, 4000, 5000, 8000, 10000, 1e6],
    'ALTI': [0,    20,   30,   40,  20,  -10,   0,   0],
    'Mveh': 55e3,
    'target_v': 72/3.6,
    'ds': 100,
    'start_location': 0,
    'travel_distance': 10000,
    'obs_horizon':2000,
    'obs_step':20,
    'acc_constraints':[-1.0,1.0],
    'speed_constraints':[1.0, 50.0],
    'timeliness_check':True
}
)
config = config.rollouts(num_rollout_workers=15, rollout_fragment_length="auto")
config = config.callbacks(CustomLoggerCallback)
config = config.resources(num_gpus=1, num_cpus_per_worker=1)
# Use to_dict() to get the old-style python config dict
# when running with tune.
tune.Tuner(  
    "PPO",
    run_config=air.RunConfig(name='PPO_train_rl_fuel', stop={"training_iteration": 3000}, verbose=3, log_to_file=True, 
        checkpoint_config=air.CheckpointConfig(
            num_to_keep = 40,
            checkpoint_frequency = 10
        )),
    param_space=config.to_dict(),
).fit()