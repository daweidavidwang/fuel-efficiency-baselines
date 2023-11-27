import numpy as np
import math
from scipy import interpolate

class Slope(object):
    def __init__(self, ALTI_X, ALTI):
        self.ALTI = ALTI
        self.ALTI_X = ALTI_X
    
    def query(self, x_node):
        h_node = []
        last_i = 0
        for x in x_node:
            for i in range(1,len(self.ALTI_X)):
                if self.ALTI_X[i] > x and x >= self.ALTI_X[i-1]:
                    h = self.ALTI[i-1] + (self.ALTI[i]-self.ALTI[i-1])*(x-self.ALTI_X[i-1])/(self.ALTI_X[i]-self.ALTI_X[i-1])
                    h_node.extend([h])
                    if x==x_node[-1]:
                        last_i = i
                    break
        
        ## add one more node for calucation
        x_node.extend([self.ALTI_X[i]])
        h_node.extend([self.ALTI[i]])

        dx_elem = np.array([x_node[i+1] - x_node[i] for i in range(0, len(x_node)-1)])
        dh_elem = np.array([h_node[i+1] - h_node[i] for i in range(0, len(h_node)-1)])
        Ge = np.arctan(dh_elem/dx_elem)
        
        return Ge
