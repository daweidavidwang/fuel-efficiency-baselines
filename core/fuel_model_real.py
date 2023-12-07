
import numpy as np
import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/nfm')
sys.path.append(os.getcwd()+'/nfm/pcc_compare')
from nfm.pcc_compare.data_loader import DataLoader
from scipy.interpolate import LinearNDInterpolator
import math
from sklearn import linear_model
import matplotlib.pyplot as plt


class FuelModel(object):
    def __init__(self, data_path):
        self.data_loader = DataLoader(data_path)
        self.data_name = data_path.split('/')[-1]
        acc = []
        vel = []
        fuel = []
        slope = []
        dis = []
        t = []
        self.height = []
        veh_weight = []

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
            veh_weight.extend([dp.vehicle_mass_kg])
        dis, t, acc, vel, fuel, slope, veh_weight = self.data_filter(dis, t, acc, vel, fuel, slope, veh_weight)
        self.interp_fuel = LinearNDInterpolator(list(zip(acc, vel, slope)), fuel)
        self.backup_fuel =linear_model.LinearRegression()
        x = []
        x.append(acc)
        x.append(vel)
        x.append(slope)
        x = np.transpose(x)
        self.dis= dis
        self.backup_fuel.fit(x, np.transpose(fuel)) #
        self.acc_range = [min(acc), max(acc)]
        self.vel_range = [min(vel), max(vel)]
        self.slope_range = [min(slope), max(slope)]
        self.vehicle_mass_kg = veh_weight
        self.slope = slope

        for idx in range(len(self.slope)):
            if idx == 0:
                self.height.extend([0.0])
            else:
                self.height.extend([self.height[-1]+self.slope[idx-1]*(self.dis[idx]-self.dis[idx-1])])

    def data_filter(self, distance, t, acc, vel, fuel, slope, weight):
        for i in range(len(fuel)):
            prod_distance = []
            prod_t = []
            prod_acc = []
            prod_vel = []
            prod_fuel = []
            prod_slope = []
            prod_weight = []
            curr_idx = i
            while curr_idx<len(distance) and distance[curr_idx]< distance[i]+50:
                prod_distance.extend([distance[curr_idx]])
                prod_acc.extend([acc[curr_idx]])
                prod_vel.extend([vel[curr_idx]])
                prod_fuel.extend([fuel[curr_idx]])
                prod_slope.extend([slope[curr_idx]])
                prod_weight.extend([weight[curr_idx]])
                curr_idx += 1
            acc[i] = np.mean(prod_acc)
            vel[i] = np.mean(prod_vel)
            fuel[i] = np.mean(prod_fuel)
            # slope[i] = np.mean(prod_slope)
            weight[i] = np.mean(prod_weight)

        return distance, t, acc, vel, fuel, slope, weight
    
    def get_map_data(self):
        return self.dis, self.height, self.slope
    
    def cal_fuel(self, a, v, grad, ds):
        fuel_rate = self.interp_fuel(a, v, grad)
        # if math.isnan(fuel_rate):
        # ## to prevent nan
        #     fuel_rate = self.backup_fuel.predict([[a, v, grad]])[0]
        #     if fuel_rate<0:
        #         fuel_rate = 0.0

        # fuel rate = l per h to g/s
        fuel_rate = fuel_rate*1000*0.83 / 3.6e3
        fuel = fuel_rate * ds / v 

        return fuel
    
    def cal_fuel_rate(self, a, v, grad):
        fuel_rate = self.interp_fuel(a, v, grad)
        # fuel rate = l per h to g/s
        fuel_rate = fuel_rate*1000*0.83 / 3.6e3

        return fuel_rate
    

    def auto_validation(self, display = False):
        def monotonicity(list):
            lin =linear_model.LinearRegression()
            lin.fit(np.arange(len(list))[:,np.newaxis], np.transpose(list))
            if lin.coef_[0]>0:
                return True
            else:
                return False
            
        a_min = self.acc_range[0]
        a_max = self.acc_range[1]
        v_min = self.vel_range[0]
        v_max = self.vel_range[1]
        slope_min = self.slope_range[0]
        slope_max = self.slope_range[1]
        a_step = (a_max - a_min)/100.0
        v_step = (v_max - v_min)/100.0
        slope_step = (slope_max - slope_min)/100.0

        acc_arange = np.arange(a_min, a_max, a_step)
        v_arange = np.arange(v_min, v_max, v_step)
        slope_arange = np.arange(slope_min, slope_max, slope_step)

        result = np.zeros([len(acc_arange), len(v_arange), len(slope_arange)])

        for ai in range(len(acc_arange)):
            for vi in range(len(v_arange)):
                for si in range(len(slope_arange)):
                    result[ai, vi, si] = self.cal_fuel_rate(acc_arange[ai], v_arange[vi], slope_arange[si])

        ## check a
        a_fail_case = []
        a_nan_case = []
        for vi in range(len(v_arange)):
            for si in range(len(slope_arange)):
                not_nan_list = []
                a_list = result[:, vi, si]
                for a in a_list:
                    if not math.isnan(a):
                        not_nan_list.extend([a])
                if len(not_nan_list)==0:
                    a_nan_case.append([vi, si])
                elif not monotonicity(not_nan_list):
                    a_fail_case.append([vi, si])

        nan_case_coordinatesx = []
        nan_case_coordinatesy = []
        a_fail_case_coordinatesx = []
        a_fail_case_coordinatesy = []
        for (vi, si) in a_nan_case:
            nan_case_coordinatesx.extend([v_arange[vi]])
            nan_case_coordinatesy.extend([slope_arange[si]])

        for (vi, si) in a_fail_case:
            a_fail_case_coordinatesx.extend([v_arange[vi]])
            a_fail_case_coordinatesy.extend([slope_arange[si]])


        if display:
            plt.scatter(nan_case_coordinatesx, nan_case_coordinatesy, marker='^')
            plt.scatter(a_fail_case_coordinatesx, a_fail_case_coordinatesy, marker='o')
            plt.savefig(self.data_name+'checka.png')
            plt.clf()

        ## check v
        v_fail_case = []
        v_nan_case = []
        for ai in range(len(acc_arange)):
            for si in range(len(slope_arange)):
                not_nan_list = []
                v_list = result[ai, :, si]
                for v in v_list:
                    if not math.isnan(v):
                        not_nan_list.extend([v])
                if len(not_nan_list)==0:
                    v_nan_case.append([ai, si])
                elif not monotonicity(not_nan_list):
                    v_fail_case.append([ai, si])

        nan_case_coordinatesx = []
        nan_case_coordinatesy = []
        v_fail_case_coordinatesx = []
        v_fail_case_coordinatesy = []
        for (ai, si) in v_nan_case:
            nan_case_coordinatesx.extend([acc_arange[ai]])
            nan_case_coordinatesy.extend([slope_arange[si]])

        for (ai, si) in v_fail_case:
            v_fail_case_coordinatesx.extend([acc_arange[ai]])
            v_fail_case_coordinatesy.extend([slope_arange[si]])


        if display:
            plt.scatter(nan_case_coordinatesx, nan_case_coordinatesy, marker='^')
            plt.scatter(v_fail_case_coordinatesx, v_fail_case_coordinatesy, marker='o')
            plt.savefig(self.data_name+'checkv.png')
            plt.clf()

        ## check slope
        slope_fail_case = []
        s_nan_case = []
        for ai in range(len(acc_arange)):
            for vi in range(len(v_arange)):
                not_nan_list = []   
                s_list = result[ai, vi, :]
                for s in s_list:
                    if not math.isnan(s):
                        not_nan_list.extend([s])
                if len(not_nan_list)==0:
                    s_nan_case.append([ai, vi])
                elif not monotonicity(not_nan_list):
                    slope_fail_case.append([ai, vi])


        nan_case_coordinatesx = []
        nan_case_coordinatesy = []
        slope_fail_case_coordinatesx = []
        slope_fail_case_coordinatesy = []
        for (ai, vi) in s_nan_case:
            nan_case_coordinatesx.extend([acc_arange[ai]])
            nan_case_coordinatesy.extend([v_arange[vi]])

        for (ai, vi) in slope_fail_case:
            slope_fail_case_coordinatesx.extend([acc_arange[ai]])
            slope_fail_case_coordinatesy.extend([v_arange[vi]])

        if display:
            plt.scatter(nan_case_coordinatesx, nan_case_coordinatesy, marker='^')
            plt.scatter(slope_fail_case_coordinatesx, slope_fail_case_coordinatesy, marker='o')
            plt.savefig(self.data_name+'checkslope.png')
            plt.clf()
            print('check finished')

        mid_vel = (self.vel_range[1] - self.vel_range[0])/2 + self.vel_range[0]
        a_border = [0.0, 0.0]
        v_border = [mid_vel, mid_vel]
        s_border = [0.0, 0.0]
        a_fixed = False
        v_fixed = False
        s_fixed = False

        while not (a_fixed and v_fixed and s_fixed):
            if not a_fixed:
                a_border = [a_border[0]-a_step, a_border[1]+a_step]
                if a_border[0]<=self.acc_range[0]:
                    a_border[0] = self.acc_range[0]
                if a_border[1]>=self.acc_range[1]:
                    a_border[1] = self.acc_range[1]
                if a_border[0] == self.acc_range[0] and a_border[1] == self.acc_range[1]:
                    a_fixed = True
            
            if not v_fixed:
                v_border = [v_border[0]-v_step, v_border[1]+v_step]
                if v_border[0] <= self.vel_range[0]:
                    v_border[0] = self.vel_range[0]
                if v_border[1] >= self.vel_range[1]:
                    v_border[1] = self.vel_range[1]
                if v_border[0] == self.vel_range[0] and v_border[1] == self.vel_range[1]:
                    v_fixed = True

            if not s_fixed:
                s_border = [s_border[0]-slope_step, s_border[1]+slope_step]
                if s_border[0] <= self.slope_range[0]:
                    s_border[0] = self.slope_range[0]
                if s_border[1] >= self.slope_range[1]:
                    s_border[1] = self.slope_range[1]
                if s_border[0] == self.slope_range[0] and s_border[1] == self.slope_range[1]:
                    s_fixed = True
            
            for (ai, vi) in s_nan_case:
                if a_border[0]< acc_arange[ai] < a_border[1] and \
                    v_border[0]< v_arange[vi]< v_border[1]:
                    a_border = [a_border[0]+a_step, a_border[1]-a_step]
                    a_fixed = True
                    v_border = [v_border[0]+v_step, v_border[1]-v_step]
                    v_fixed = True
            for (ai, vi) in slope_fail_case:
                if a_border[0]< acc_arange[ai] < a_border[1] and \
                    v_border[0]< v_arange[vi]< v_border[1]:
                    a_border = [a_border[0]+a_step, a_border[1]-a_step]
                    a_fixed = True
                    v_border = [v_border[0]+v_step, v_border[1]-v_step]
                    v_fixed = True

            for (ai, si) in v_nan_case:
                if a_border[0]< acc_arange[ai] < a_border[1] and \
                    s_border[0]< slope_arange[si] < s_border[1]:
                    a_border = [a_border[0]+a_step, a_border[1]-a_step]
                    a_fixed = True
                    s_border = [s_border[0]+slope_step, s_border[1]-slope_step]
                    s_fixed = True
            for (ai, si) in v_fail_case:
                if a_border[0]< acc_arange[ai] < a_border[1] and \
                    s_border[0]< slope_arange[si] < s_border[1]:
                    a_border = [a_border[0]+a_step, a_border[1]-a_step]
                    a_fixed = True
                    s_border = [s_border[0]+slope_step, s_border[1]-slope_step]
                    s_fixed = True

            for (vi, si) in a_nan_case:
                if s_border[0]< slope_arange[si] < s_border[1] and \
                    v_border[0]< v_arange[vi]< v_border[1]:
                    s_border = [s_border[0]+slope_step, s_border[1]-slope_step]
                    s_fixed = True
                    v_border = [v_border[0]+v_step, v_border[1]-v_step]
                    v_fixed = True
            for (vi, si) in a_fail_case:
                if s_border[0]< slope_arange[si] < s_border[1] and \
                    v_border[0]< v_arange[vi]< v_border[1]:
                    s_border = [s_border[0]+slope_step, s_border[1]-slope_step]
                    s_fixed = True
                    v_border = [v_border[0]+v_step, v_border[1]-v_step]
                    v_fixed = True

        print(v_border)
        print(a_border)
        print(s_border)
        self.v_border = v_border
        self.a_border = a_border
        self.s_border = s_border

if __name__ == "__main__":
    data_path = '/home/dawei/Documents/pkl_data/2d070af5-e7fd-4214-9968-69bd5a4643cb.pkl'
    fmr = FuelModel(data_path)
    fmr.auto_validation(True)