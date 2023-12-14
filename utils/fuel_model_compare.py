from core.fuel_model import FuelModel as FM
from core.fuel_model_real import FuelModel as FMR
import numpy as np
import matplotlib.pyplot as plt
import math

path = '/home/dawei/Downloads/content_1701161245375.pkl'
fmr = FMR(path)
fm = FM(49e3)

fm_fuel = []
fmr_fuel = []
real_data_fuel = []
ds = 100
accl = []
vell = []
fuell = []
slopel = []
for i in range(14000):
    dp = fmr.data_loader.GetTBasedData(i)
    acc = dp.acc
    vel = dp.speed
    fuel = dp.fuel_lph
    grad_rad = dp.slope_rad
    accl.extend([dp.acc])
    vell.extend([dp.speed])
    fuell.extend([dp.fuel_lph])
    slopel.extend([dp.slope_rad])
    real_data_fuel.extend([(fuel*1000*0.83 / 3.6e3)* ds / vel ])
    fm_fuel.extend([fm.cal_fuel(acc, vel, grad_rad, ds)])
    fmr_fuel.extend([fmr.cal_fuel(acc, vel, grad_rad, ds)])

real_data_fuel = np.array(real_data_fuel)
fm_fuel = np.array(fm_fuel)
fmr_fuel = np.array(fmr_fuel)
error =  np.power(fm_fuel-real_data_fuel, 2)
large_fail_case = []
for i in range(len(error)):
    if math.sqrt(error[i]) > fm_fuel[i]:
        large_fail_case.extend([(accl[i], vell[i], slopel[i])])
np.sum(np.power(fmr_fuel-real_data_fuel, 2))

