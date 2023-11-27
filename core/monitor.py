from matplotlib import pyplot as plt
import numpy as np 
class Monitor(object):
    def __init__(self):
        self.data_record = dict()
        self.record_keywords = ['t', 'speed', 'reward', 'total_fuel', 'Ge']
        for kw in self.record_keywords:
            self.data_record[kw] = []

    def step(self, env, reward, ge):
        self.data_record['t'].extend([env.env_step])
        self.data_record['speed'].extend([env.veh.velocity])
        self.data_record['reward'].extend([reward])
        self.data_record['total_fuel'].extend([env.veh.total_fuel])
        self.data_record['Ge'].extend([ge])
    
    def plot(self):
        fig, (ax)= plt.subplots(1, 1)
        ax1 = ax.twinx()

        line,  = ax1.plot(np.array(self.data_record['Ge']), label='Ge')
        line,  = ax.plot(np.array(self.data_record['speed']), label='speed')
        ax.set_xlabel('step')
        ax.legend()
        plt.show()