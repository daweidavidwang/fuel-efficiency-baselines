import numpy as np
import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/nfm')
sys.path.append(os.getcwd()+'/nfm/pcc_compare')
from nfm.pcc_compare.data_loader import DataLoader
from scipy.interpolate import LinearNDInterpolator
import math
from sklearn import linear_model

class MissionLoader(object):
    def __init__(self, data_path):
        self.data_loader = DataLoader(data_path)
        self.x = []
        self.height = []
        self.slope = []

        for i in range(self.data_loader.moment_total_length):
            dp = self.data_loader.GetTBasedData(i)
            self.x.extend([dp.total_distance_m])
            self.height.extend([dp.altitude])
            self.slope.extend([dp.slope_rad])

    def get_map_data(self):
        return self.x, self.height, self.slope
        