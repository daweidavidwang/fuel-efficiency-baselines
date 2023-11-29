
import numpy as np
import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/nfm')
sys.path.append(os.getcwd()+'/nfm/pcc_compare')
from nfm.pcc_compare.data_loader import DataLoader
from scipy.interpolate import LinearNDInterpolator
import math
from sklearn import linear_model

class FuelModel(object):
    def __init__(self, data_path):
        self.data_loader = DataLoader(data_path)
        acc = []
        vel = []
        fuel = []
        slope = []
        dis = []
        t = []

        for i in range(self.data_loader.moment_total_length):
            dp = self.data_loader.GetTBasedData(i)
            if dp.is_brake or dp.gear_position != 12:
                continue
            dis.extend([dp.total_distance_m])
            t.extend([dp.timestamp_sec])
            acc.extend([dp.acc])
            vel.extend([dp.speed])
            fuel.extend([dp.fuel_lph])
            slope.extend([dp.slope_rad])
        dis, t, acc, vel, fuel, slope = self.data_filter(dis, t, acc, vel, fuel, slope)
        self.interp_fuel = LinearNDInterpolator(list(zip(acc, vel, slope)), fuel)
        self.backup_fuel =linear_model.LinearRegression()
        x = []
        x.append(acc)
        x.append(vel)
        x.append(slope)
        x = np.transpose(x)
        self.backup_fuel.fit(x, np.transpose(fuel)) #
        self.acc_range = [min(acc), max(acc)]
        self.vel_range = [min(vel), max(vel)]
        self.slope_range = [min(slope), max(slope)]

    def data_filter(self, distance, t, acc, vel, fuel, slope):

        for i in range(len(fuel)):
            prod_distance = []
            prod_t = []
            prod_acc = []
            prod_vel = []
            prod_fuel = []
            prod_slope = []
            curr_idx = i
            while curr_idx<len(distance) and distance[curr_idx]< distance[i]+50:
                prod_distance.extend([distance[curr_idx]])
                prod_acc.extend([acc[curr_idx]])
                prod_vel.extend([vel[curr_idx]])
                prod_fuel.extend([fuel[curr_idx]])
                prod_slope.extend([slope[curr_idx]])
                curr_idx += 1
            acc[i] = np.mean(prod_acc)
            vel[i] = np.mean(prod_vel)
            fuel[i] = np.mean(prod_fuel)
            slope[i] = np.mean(prod_slope)

        return distance, t, acc, vel, fuel, slope

    def cal_fuel(self, a, v, grad, ds):
        fuel_rate = self.interp_fuel(a, v, grad)
        if math.isnan(fuel_rate):
        ## to prevent nan
            fuel_rate = self.backup_fuel.predict([[a, v, grad]])[0]
            if fuel_rate<0:
                fuel_rate = 0.0

        # fuel rate = l per h to g/s
        fuel_rate = fuel_rate*1000*0.83 / 3.6e3
        fuel = fuel_rate * ds / v 

        return fuel
    
    def cal_fuel_rate(self, a, v, grad):
        fuel_rate = self.interp_fuel(a, v, grad)
        # fuel rate = l per h to g/s
        fuel_rate = fuel_rate*1000*0.83 / 3.6e3

        return fuel_rate