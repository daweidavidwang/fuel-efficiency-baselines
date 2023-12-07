import numpy as np
import math
from scipy import interpolate

class Slope(object):
    def __init__(self):
        self.mode = None
        self.slope = None
        self.ALTI_X = None
        self.ALTI = None
        self.slope_range = [0.0,0.0]
    
    def construct(self, mode, ALTI_X, input_b, slope_range):
        ## mode=slope, input_b = slope; height, input_b = height
        if mode == 'slope':
            self.slope = input_b
            self.ALTI_X = ALTI_X
        elif mode == 'height':
            self.ALTI = input_b
            self.ALTI_X = ALTI_X
        else:
            print('error in mode selection, you can choose slope or height')
        self.mode = mode
        self.slope_range = slope_range
        return 

    def _binary_search(self, array, target):
        ## find the index idx and jdx that array[idx]<= target < array[jdx]
        def search_func(array, target, idx, jdx):
            if jdx == idx+1:
                return idx, jdx
            mid = idx + int((jdx - idx) / 2)
            if target == array[mid]:
                return mid, mid+1
            elif target < array[mid]:
                return search_func(array, target, idx, mid)
            else: 
                #target > array[mid]:
                return search_func(array, target, mid, jdx)

        return search_func(array, target, 0, len(array)-1)

    def query(self, x_node):
        if self.mode == 'slope':
            slope = []
            for x in x_node:
                idx, jdx = self._binary_search(self.ALTI_X, x)
                s = (self.slope[idx]+self.slope[jdx])/2
                if s < self.slope_range[0]:
                    s = self.slope_range[0]
                elif s > self.slope_range[1]:
                    s = self.slope_range[1]
                slope.extend([s])
            return np.array(slope)

        elif self.mode == 'height':
            h_node = []
            last_i = 0
            for x in x_node:
                idx, jdx = self._binary_search(self.ALTI_X, x)
                h = self.ALTI[idx] + (self.ALTI[jdx]-self.ALTI[idx])*(x-self.ALTI_X[idx])/(self.ALTI_X[jdx]-self.ALTI_X[idx])
                h_node.extend([h])
                if x==x_node[-1]:
                    last_i = jdx
            
            ## add one more node for calucation
            x_node.extend([self.ALTI_X[last_i]])
            h_node.extend([self.ALTI[last_i]])

            dx_elem = np.array([x_node[i+1] - x_node[i] for i in range(0, len(x_node)-1)])
            dh_elem = np.array([h_node[i+1] - h_node[i] for i in range(0, len(h_node)-1)])
            Ge = np.arctan(dh_elem/dx_elem)
            for i in range(len(Ge)):
                if math.isnan(Ge[i]):
                    ## prevent nan in Ge
                    Ge[i] = 0.0

            return Ge
        else: 
            print('incorrect mode setup')
            return []