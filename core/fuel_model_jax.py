
import jax.numpy as np
import jax

class FuelModel(object):
    def __init__(self, Mveh):
        self.cd = 0.398125
        self.front_area = 9.773749
        self.mu = 0.012
        self.g = 9.801
        self.Mveh = Mveh
        self.gb_eff = 0.931 ## driveline efficiency
        self.k_d = 0.2797 * 3.6 * 3.6
        self.mu_s = 0.0041
        self.mu_d = 0.00001517 * 3.6
        self.coefs = [160.4, 0.617, -0.006015, 1.732e-5] ##bsfc map
        self.loss_p = 0 ## engine loss power (const)

    def cal_fuel(self, a, v, grad, ds):
        wind = v* self.k_d * v**2
        rolling_stable = v* self.Mveh * self.g * self.mu_s
        rolling_weight = v* self.Mveh * self.g * np.sin(grad)
        rolling_running = v* self.Mveh * self.g * self.mu_d * v
        Fac = self.Mveh * a
        pe = self.loss_p + (wind + rolling_stable + rolling_running + rolling_weight + Fac)/(1000*self.gb_eff)
        pe = jax.nn.relu(pe)
        # jax.debug.print("wind:{}", wind)
        # jax.debug.print("rolling_stable:{}", rolling_stable)
        # jax.debug.print("rolling_weight:{}", rolling_weight)
        # jax.debug.print("rolling_running:{}", rolling_running)
        # jax.debug.print("Fac:{}", Fac)
        fuel_rate = 0
        for i in range(len(self.coefs)):
            fuel_rate += self.coefs[i] * pow(pe, i)

        fuel_rate = pe*fuel_rate
        # The fuel rate for IPOPT solver lacks a *pe 
        fuel_rate = jax.nn.relu(fuel_rate)

        
        fuel = fuel_rate * ds / v / 3.6e3
        # jax.debug.print("fc:{}", fuel)


        return fuel