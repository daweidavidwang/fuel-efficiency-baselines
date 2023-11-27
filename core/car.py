import math
class Car(object):
    def __init__(self, veh_config):
        self._step = 0
        self.total_fuel = 0
        self.position = veh_config['start_location']
        self.slope_map = veh_config['slope_map']
        self.fuel_estimator = veh_config['fuel_estimator']
        self.velocity = veh_config['velocity']
        self.ds = veh_config['ds']
        self.max_speed = veh_config['speed_constraints'][1]
        self.min_speed = veh_config['speed_constraints'][0]
        self.total_running_time = 0

    def boundary_check(self, value, max, min):
        if value > max:
            return max
        elif value< min:
            return min 
        else:
            return value

    def compute_travel_time(self, acceleration):
        if acceleration==0:
            t=self.ds/self.velocity
            return t, acceleration, self.velocity

        elif self.velocity <= self.min_speed and acceleration<0:
            t=self.ds/self.velocity
            return t, acceleration, self.velocity

        elif self.velocity>= self.max_speed and acceleration>0:
            t=self.ds/self.velocity
            return t, acceleration, self.velocity

        else:
            if 2*acceleration*self.ds + self.velocity**2>=0:
                ## final speed exists, final speed = v^2 + 2as
                final_velocity  = math.sqrt(2*acceleration*self.ds + self.velocity**2)
                if self.max_speed > final_velocity and final_velocity > self.min_speed:
                    t = (final_velocity-self.velocity)/acceleration
                    return t, acceleration, final_velocity
                else:
                    # new velocity violates the speed constraints
                    final_velocity = max(min(self.max_speed, final_velocity), self.min_speed)
                    acc = (final_velocity**2 - self.velocity**2)/(2*self.ds)
                    t = (final_velocity-self.velocity)/acc
                    return t, acc, final_velocity
            else:
                print('error in calucating final speed, not exist')
                t=self.ds/self.velocity
                return t, acceleration, self.velocity

    def step(self, acceleration):
        ## compute travel time, acc(if original acc is not feasible) final velocity
        dt, acc, final_velocity = self.compute_travel_time(acceleration)

        self.position += self.ds
        self.velocity = final_velocity
        slope = self.slope_map.query([self.position])
        fuel = self.fuel_estimator.cal_fuel(acc, self.velocity, slope[0], self.ds)
        self.total_fuel += fuel
        self._step += 1        
        self.total_running_time += dt
        return self.position, self.velocity, fuel, self.total_fuel, slope

    def get_avg_speed_past(self):
        return self.position/self.total_running_time if self.total_running_time else 0

