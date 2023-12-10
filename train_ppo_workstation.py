import ray 
from env import Env
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from core.custom_logger import CustomLoggerCallback
from ray import tune
from core.fuel_model_real import FuelModel
from core.slope_map import Slope
from ray.tune.stopper import TrialPlateauStopper, MaximumIterationStopper

fuel_model_data_path = '2d070af5-e7fd-4214-9968-69bd5a4643cb.pkl'
fuel_model =FuelModel(fuel_model_data_path)

# slope_map = Slope()
# ALTI_X =  [0, 2000, 3000, 4000, 5000, 8000, 10000, 1e6]
# ALTI =  [0,    6,   9,   12,  9,  0,   -3,   0]
# slope_map.construct(mode='height', ALTI_X=ALTI_X, input_b=ALTI, slope_range=[-0.025183296730539186, 0.025183296730539186] )


## real world map
ALTI_X, ALTI, ALTI_slope = fuel_model.get_map_data()
slope_map = Slope()
slope_map.construct(mode='slope', ALTI_X=ALTI_X, input_b=ALTI_slope, slope_range=[-0.005093949143751466, 0.005093949143751466] )


# ray.init(num_gpus=1, num_cpus=16)
ray.init(local_mode=True)
config = PPOConfig()
# Update the config object.
# config = config.training(lr=tune.grid_search([0.001, 0.0001]))
# config = config.training(lr=1e-4, train_batch_size = 2048*64*2, sgd_minibatch_size=2048*32*2)
config = config.training(lr=1e-4, train_batch_size = 2048*64*4, sgd_minibatch_size=2048*32*4)
# lr_schedule=[[0, 1e-4], [20000000, 1e-5]]
# config = config.training(lr=tune.grid_search([ 1e-6,  5e-6, 1e-3, 1e-4, 1e-5]), train_batch_size = 2048*64*2, sgd_minibatch_size=2048*32*2)

# Set the config object's env.
config = config.environment(env=Env,env_config={
    'slope':slope_map,
    'Mveh': 55e3,
    'target_v': 21.0,
    'start_velocity_range': [19, 24],
    'ds': 100,
    'start_location': 0,
    'start_location_range': [int(ALTI_X[0]),int(ALTI_X[-1])-11000],
    'travel_distance': 10000,
    'travel_distance_range': [5000, 10000],
    'obs_horizon':2000,
    'obs_step':20,
    'acc_constraints':[-0.02, 0.02] ,
    'speed_constraints':[18.922531942120045, 24.616176171879957],
    'timeliness_check':True,
    'fuel_model':fuel_model,
    'slope_range':fuel_model.slope_range
}
)
# def stop_fn(trial_id: str, result: dict) -> bool:
#     print(str(result))
#     return result["episode_reward_mean"]

config = config.rollouts(num_rollout_workers=1, rollout_fragment_length="auto")
config = config.callbacks(CustomLoggerCallback)
config = config.resources(num_gpus=1, num_cpus_per_worker=1)
# Use to_dict() to get the old-style python config dict
# when running with tune.
tune.Tuner(  
    "PPO",
    run_config=air.RunConfig(name='PPO_train_rl_fuel_realdata', \
    stop={"training_iteration": 500}, verbose=3, log_to_file=True, 
        checkpoint_config=air.CheckpointConfig(
            num_to_keep = 40,
            checkpoint_frequency = 5
        ),
        ),
    param_space=config.to_dict(),
).fit()



# stop=[TrialPlateauStopper(metric='episode_reward_mean', std=5, num_results=10), MaximumIterationStopper(max_iter=3000)]