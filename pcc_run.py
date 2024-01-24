from env import Env

from pcc_ipopt.pcc import PCC_alg
from core.fuel_model import FuelModel as FM
from core.slope_map import Slope

fuel_model_data_path ='/home/dawei/Documents/pkl_data/2d070af5-e7fd-4214-9968-69bd5a4643cb.pkl'
fuel_model = FMR(fuel_model_data_path)
fuel_model_fake = FM(Mveh=55e3)
# ## real world map
ALTI_X, ALTI, ALTI_slope = fuel_model.get_map_data(factor=0,downsampling=1000)
slope_map = Slope()
# slope_map.construct(mode='slope', ALTI_X=ALTI_X, input_b=ALTI_slope, slope_range=[-0.005093949143751466, 0.005093949143751466] )
slope_map.construct(mode='height', ALTI_X=ALTI_X, input_b=ALTI, slope_range=[-0.005093949143751466, 0.005093949143751466] )

# slope_map = Slope()
# ALTI_X =  [0, 2000, 3000, 4000, 5000, 8000, 10000, 1e6]
# ALTI =  [0,    -6,   -9,   -12,  -9,  0,   3,   0]
# slope_map.construct(mode='height', ALTI_X=ALTI_X, input_b=ALTI, slope_range=[-0.005093949143751466, 0.005093949143751466])

env_config = {
    'slope':slope_map,
    'Mveh': 24779,
    'target_v': 21.0,
    'ds': 100,
    'start_location': 120000,
    'travel_distance': 10000,
    'obs_horizon':2000,
    'obs_step':20,
    'acc_constraints':[-0.1, 0.1] ,
    'speed_constraints':[18.922531942120045, 24.616176171879957],
    'fuel_model':fuel_model
}
pcc_config = {
    'ALTI_X': ALTI_X,
    'ALTI': ALTI,
    'Mveh': 55e3,
    'target_v': 21.0,
    'ds': 100,
    'start_location': 120000,
    'travel_distance': 10000,
    'obs_horizon':2000,
    'obs_step':20,
    'acc_constraints':[-0.1, 0.1] ,
    'speed_constraints':[18.922531942120045, 24.616176171879957],
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
# print(reward_list)
# print(action_list)
print(info['total_fuel'])
print(info['const_baseline_fuel'])