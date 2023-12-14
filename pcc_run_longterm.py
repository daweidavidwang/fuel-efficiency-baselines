from env import Env

from pcc_ipopt.pcc import PCC_alg
from core.fuel_model_real import FuelModel as FMR
from core.fuel_model import FuelModel as FM
from core.slope_map import Slope
from matplotlib import pyplot as plt
import numpy as np

fuel_model_data_path ='/home/dawei/Documents/pkl_data/2d070af5-e7fd-4214-9968-69bd5a4643cb.pkl'
fuel_model = FMR(fuel_model_data_path)

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
    'timeliness_check':True,
    'fuel_model':fuel_model
}
reward_list = []
obs_list = []
action_list = []
speed = []
slope = []
height = []
total_fuel = 0
total_baseline_fuel = 0
start_velocity = 21.0
prev_height_end = 0.0
env = Env(env_config)
for start_loc in range(120000,220000, 10000):
    obs, info = env.reset(options={
        'start_location':start_loc,
        "start_velocity":start_velocity
    })
    pcc_config = {
        'ALTI_X': ALTI_X,
        'ALTI': ALTI,
        'Mveh': 24779,
        'target_v': 21.0,
        'ds': 100,
        'start_location': start_loc,
        'travel_distance': 10000,
        'obs_horizon':2000,
        'obs_step':20,
        'acc_constraints':[-0.1, 0.1] ,
        'speed_constraints':[18.922531942120045, 24.616176171879957],
    }

    pcc = PCC_alg(pcc_config)
    res = pcc.solve()

    if res.success:
        v_list = list(res.x)
    else:
        print("pcc cannot find optimal solution!")
        v_list = list(res.x)
    # v_list.extend([v_list[-1]])
    reward_list = []
    obs_list = []
    action_list = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        curr_step = env.env_step
        action = (v_list[curr_step+1]**2 - v_list[curr_step]**2)/(2*env.ds)
        action_list.extend([action])
        obs, reward, terminated, truncated, info = env.step(action)
        reward_list.extend([reward])
    speed.extend(env.monitor.data_record['speed'])
    record_h = np.array(env.monitor.data_record['height']) + prev_height_end
    prev_height_end = record_h[-1]
    height.extend(record_h)
    slope.extend(env.monitor.data_record['Ge'])
    total_fuel += info['total_fuel']
    total_baseline_fuel += info['const_baseline_fuel']
    start_velocity = env.monitor.data_record['speed'][-1]

fig, (ax)= plt.subplots(1, 1)
ax1 = ax.twinx()
line,  = ax1.plot(np.array(height), label='height',  color='red')
line,  = ax.plot(np.array(speed), label='speed', color='blue')
ax.set_xlabel('step')
ax.set_ylabel('Speed(m/s)')
ax1.set_ylabel('height (m)')
ax.legend()
ax.set_title('PCC Total Fuel='+str(total_fuel))
print(total_fuel)
print(total_baseline_fuel)
plt.show()