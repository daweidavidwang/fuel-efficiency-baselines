import sys, os
sys.path.append(os.getcwd())
import os
from core.fuel_model_real import FuelModel as FMR
import matplotlib.pyplot as plt
import numpy as np

path = '/home/dawei/Documents/pkl_data/'
file_list = os.listdir(path)

file_list = [
    '000f837b-e114-48b1-b176-0c449d11c39e.pkl',
    '2d070af5-e7fd-4214-9968-69bd5a4643cb.pkl',
    '3e307090-b318-4b82-a655-41e1484ccdea.pkl',
    '9f248a4b-a571-425f-b16b-9f9599e4ca13.pkl',
    '9094158d-f57e-489c-95fd-6911e7c3fbd4.pkl',
    '99fabe36-8c1a-4276-8357-410f119304a4.pkl'
]

v = 20.0
a = 0.1
slope = 0.0
weight = []
fuel = []

result = []
for file in file_list:
    
    fuel_model = FMR(path+file)
    fuel_model.auto_validation(False)
    weight.extend([np.mean(fuel_model.vehicle_mass_kg)])
    fuel.extend([fuel_model.cal_fuel_rate(a, v, slope)])
    result.extend([(weight, fuel)])
plt.plot(weight, fuel)
    # fig, (ax)= plt.subplots(1, 1)
    # ax1 = ax.twinx()
    # avg_weight = np.mean(fuel_model.vehicle_mass_kg)
    # line,  = ax1.plot(np.array(fuel_model.vehicle_mass_kg), label='Mass',  color='red')
    # line,  = ax.plot(np.array(fuel_model.slope), label='Slope', color='blue')
    # ax.set_xlabel('step')
    # ax1.set_ylabel('Mass (kg)')
    # ax.set_ylabel('Slope (rad)')
    # ax.legend()
    # ax.set_title(file)
    # plt.savefig(path+file+'.png')
    # plt.clf()
    # print('data:'+file+' weight:'+str(avg_weight)+' slope range:'+str(fuel_model.s_border)+' acc range:'+str(fuel_model.a_border)+' veh range:'+str(fuel_model.v_border))

