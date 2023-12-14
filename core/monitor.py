from matplotlib import pyplot as plt
import numpy as np 
class Monitor(object):
    def __init__(self):
        self.data_record = dict()
        self.record_keywords = ['t', 'speed', 'reward', 'total_fuel', 'Ge', 'height']
        for kw in self.record_keywords:
            self.data_record[kw] = []

    def step(self, env, reward, ge):
        self.data_record['t'].extend([env.env_step])
        self.data_record['speed'].extend([env.veh.velocity])
        self.data_record['reward'].extend([reward])
        self.data_record['total_fuel'].extend([env.veh.total_fuel])
        self.data_record['Ge'].extend(ge)
        if len(self.data_record['height'])==0:
            self.data_record['height'].extend([0.0])
        else:
            self.data_record['height'].extend(self.data_record['height'][-1]+ge*env.ds)
    
    def plot(self):
        fig, (ax)= plt.subplots(1, 1)
        ax1 = ax.twinx()

        line,  = ax1.plot(np.array(self.data_record['height']), label='height',  color='red')
        line,  = ax.plot(np.array(self.data_record['speed']), label='speed', color='blue')
        ax.set_xlabel('step')
        ax.set_ylabel('Speed(m/s)')
        ax1.set_ylabel('height (m)')
        ax.legend()
        ax.set_title('Totoal Fuel='+str(self.data_record['total_fuel'][-1]))
        plt.show()