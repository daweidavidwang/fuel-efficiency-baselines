from core.fuel_model import FuelModel as FM
from core.fuel_model_real import FuelModel as FMR
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import linear_model

def monotonicity(list):
    lin =linear_model.LinearRegression()
    lin.fit(np.arange(len(list))[:,np.newaxis], np.transpose(list))
    if lin.coef_[0]>0:
        return True
    else:
        return False
    
fuel_model_data_path = '/home/dawei/Downloads/0695e99d-a2ca-45d4-ace0-ab16e84cd94c.pkl'
fmr = FMR(fuel_model_data_path)

ds = 100

a_min = fmr.acc_range[0]
a_max = fmr.acc_range[1]
v_min = fmr.vel_range[0]
v_max = fmr.vel_range[1]
slope_min = fmr.slope_range[0]
slope_max = fmr.slope_range[1]
a_step = (a_max - a_min)/20.0
v_step = (v_max - v_min)/20.0
slope_step = (slope_max - slope_min)/20.0

acc_arange = np.arange(a_min, a_max, a_step)
v_arange = np.arange(v_min, v_max, v_step)
slope_arange = np.arange(slope_min, slope_max, slope_step)

result = np.zeros([len(acc_arange), len(v_arange), len(slope_arange)])

for ai in range(len(acc_arange)):
    for vi in range(len(v_arange)):
        for si in range(len(slope_arange)):
            result[ai, vi, si] = fmr.cal_fuel_rate(acc_arange[ai], v_arange[vi], slope_arange[si])

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



plt.scatter(nan_case_coordinatesx, nan_case_coordinatesy, marker='^')
plt.scatter(a_fail_case_coordinatesx, a_fail_case_coordinatesy, marker='o')
plt.savefig('checka.png')

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
a_fail_case_coordinatesx = []
a_fail_case_coordinatesy = []
for (ai, si) in v_nan_case:
    nan_case_coordinatesx.extend([acc_arange[ai]])
    nan_case_coordinatesy.extend([slope_arange[si]])

for (ai, si) in a_fail_case:
    a_fail_case_coordinatesx.extend([acc_arange[vi]])
    a_fail_case_coordinatesy.extend([slope_arange[si]])



plt.scatter(nan_case_coordinatesx, nan_case_coordinatesy, marker='^')
plt.scatter(a_fail_case_coordinatesx, a_fail_case_coordinatesy, marker='o')
plt.savefig('checkv.png')
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
a_fail_case_coordinatesx = []
a_fail_case_coordinatesy = []
for (ai, vi) in s_nan_case:
    nan_case_coordinatesx.extend([acc_arange[ai]])
    nan_case_coordinatesy.extend([v_arange[vi]])

for (ai, vi) in a_fail_case:
    a_fail_case_coordinatesx.extend([acc_arange[vi]])
    a_fail_case_coordinatesy.extend([v_arange[vi]])

plt.scatter(nan_case_coordinatesx, nan_case_coordinatesy, marker='^')
plt.scatter(a_fail_case_coordinatesx, a_fail_case_coordinatesy, marker='o')
plt.savefig('checkslope.png')
plt.clf()
print('check finished')

a_border = [0.0, 0.0]
v_border = [15.0, 15.0]
s_border = [0.0, 0.0]
a_fixed = False
v_fixed = False
s_fixed = False

while not (a_fixed and v_fixed and s_fixed):
    if not a_fixed:
        a_border = [a_border[0]-a_step, a_border[1]+a_step]
    
    if not v_fixed:
        v_border = [v_border[0]-v_step, v_border[1]+v_step]

    if not s_fixed:
        s_border = [s_border[0]-slope_step, s_border[1]+slope_step]
    
    for (ai, vi) in s_nan_case:
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

    for (vi, si) in a_nan_case:
        if s_border[0]< slope_arange[si] < s_border[1] and \
            v_border[0]< v_arange[vi]< v_border[1]:
            s_border = [s_border[0]+slope_step, s_border[1]-slope_step]
            s_fixed = True
            v_border = [v_border[0]+v_step, v_border[1]-v_step]
            v_fixed = True

print(v_border)
print(a_border)
print(s_border)