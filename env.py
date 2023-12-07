import gymnasium as gym
from core.car import Car


import numpy as np 
from gymnasium.spaces import Box
import math
import random 
from core.monitor import Monitor
EPSILON = 0.000001


class Env(gym.Env):
    def __init__(self, env_config=None):
        super().__init__()
        self._step = 0
        self.env_config = env_config
        self.timeliness_check = env_config['timeliness_check'] if 'timeliness_check' in env_config else False

        self.slope_map = env_config['slope']
        self.slope_range = env_config['slope_range'] if 'slope_range' in env_config else [-0.25, 0.25]
        self.Mveh = env_config['Mveh']
        self.fuel_model = env_config['fuel_model']

        self.target_v = env_config['target_v']  ## m/s
        self.ds = env_config['ds'] ## travel distance for every step (m)
        self.start_location = env_config['start_location'] if 'start_location' in env_config else 0.0
        self.start_location_range = env_config['start_location_range'] if 'start_location_range' in env_config else None

        self.travel_distance = env_config['travel_distance']
        self.end_location = self.start_location + self.travel_distance
        self.max_travel_time = self.travel_distance/self.target_v 
        self.veh = Car(veh_config={
            'start_location':self.start_location,
            'slope_map':self.slope_map,
            'fuel_estimator':self.fuel_model,
            'velocity':self.target_v,
            'ds': self.ds,
            'speed_constraints': env_config['speed_constraints']
        })

        self.veh_const = Car(veh_config={
            'start_location':self.start_location,
            'slope_map':self.slope_map,
            'fuel_estimator':self.fuel_model,
            'velocity':self.target_v,
            'ds': self.ds,
            'speed_constraints': env_config['speed_constraints']
        })

        self.obs_horizon = env_config['obs_horizon'] #forward distance that agent can observe
        self.obs_step = env_config['obs_step'] # len(obs) = obs_horizon/obs_step

        self.acc_constraints = env_config['acc_constraints'] # [min_acc, max_acc]
        
        self.monitor = Monitor()
    @property
    def action_space(self):
        return Box(
            low=self.acc_constraints[0],
            high=self.acc_constraints[1],
            shape=(1, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        return Box(
            low=-1,
            high=1,
            shape=(int(self.obs_horizon/self.obs_step)+3, ),
            dtype=np.float32)
    
    @property
    def env_step(self):
        return self._step

    def timeliness(self):
        remaining_t = self.max_travel_time - self.veh.total_running_time
        if self.end_location - self.veh.position >= self.veh.velocity*remaining_t + 0.5*self.acc_constraints[1]*remaining_t**2:
            return True
        else:
            return False

    def get_ahead_slope(self):
        x = [self.veh.position+i*self.obs_step for i in range(int(self.obs_horizon/self.obs_step))]
        max_slope_boundary = max(abs(self.slope_range[0]), self.slope_range[1])
        slope = self.slope_map.query(x) * (1/max_slope_boundary)
        return slope

    def get_obs(self):
        past_avg_speed = self.veh.get_avg_speed_past()
        past_speed_est = self.boundary_check((self.target_v-past_avg_speed)/self.target_v, 1, -1) ## past speed est
        remaining_distance  =  self.boundary_check((self.end_location-self.veh.position)/self.travel_distance, 1, -1) ## remaining distance
        remaining_time =  self.boundary_check((self.max_travel_time-self.veh.total_running_time)/(self.max_travel_time), 1, -1) ## remaining time
        return past_speed_est, remaining_distance, remaining_time

    def boundary_check(self, value, max, min):
        if value > max:
            return max
        elif value < min:
            return min 
        else:
            return value

    def compute_reward(self, reward_param={}):
        fuel = reward_param['fuel']
        acc_pow2 = reward_param['acc']
        return -fuel 

    def reset(self, seed=None, options=None):
        if self.start_location_range:
            self.start_location = random.randint(self.start_location_range[0], self.start_location_range[1])
            self.end_location = self.start_location + self.travel_distance

        self.veh = Car(veh_config={
            'start_location':self.start_location,
            'slope_map':self.slope_map,
            'fuel_estimator':self.fuel_model,
            'velocity':self.target_v,
            'ds': self.ds,
            'speed_constraints': self.env_config['speed_constraints']
        })
        self.veh_const = Car(veh_config={
            'start_location':self.start_location,
            'slope_map':self.slope_map,
            'fuel_estimator':self.fuel_model,
            'velocity':self.target_v,
            'ds': self.ds,
            'speed_constraints': self.env_config['speed_constraints']
        })
        self._step = 0

        obs = self.get_ahead_slope()
        past_speed_est, remaining_distance, remaining_time = self.get_obs()
        obs = np.append(obs, [past_speed_est, remaining_distance, remaining_time])
        info = {
            'total_fuel':0
        }
        return obs, info

    def step(self, action):
        if not isinstance(action, float):
            action = action[0]
        action = action* self.acc_constraints[1]
        if self.timeliness_check and self.timeliness():
            ## risk of timeout
            action = self.acc_constraints[1]
        position, velocity, fuel, total_fuel, slope = self.veh.step(action)
        _, _, _, total_fuel_const, _ = self.veh_const.step(0.0)
        obs = self.get_ahead_slope()
        past_speed_est, remaining_distance, remaining_time = self.get_obs()
        obs = np.append(obs, [past_speed_est, remaining_distance, remaining_time])

        info = {
            'total_fuel':total_fuel,
            'const_baseline_fuel':total_fuel_const
        }
        reward_param = {
            'fuel': fuel,
            'acc': action**2
        }
        reward = self.compute_reward(reward_param)

        if position >= self.end_location:
            terminated = True
            reward =  - total_fuel
        else:
            terminated = False
        truncated = False
        # if self.veh.total_running_time> self.max_travel_time:
        #     truncated = True
        #     reward =  - total_fuel
        # else:
        #     truncated = False
        self.monitor.step(self, reward, slope)
        self._step += 1

        return obs, reward, terminated, truncated, info


