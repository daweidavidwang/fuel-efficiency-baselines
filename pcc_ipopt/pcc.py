import sys, os
sys.path.append(os.getcwd())
from jax.config import config
# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)
# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update('jax_platform_name', 'cpu')
import math
import jax
import jax.numpy as np
from jax.lax import cond
from jax import jit, grad, jacfwd, jacrev
from cyipopt import minimize_ipopt
from core.fuel_model_jax import FuelModel

class PCC_alg(object):
    def __init__(self, env_config):
        self.v0 = self.target_v = env_config['target_v']
        if 'start_velocity' in env_config:
            self.v0 = env_config['start_velocity'] 
        self.ds = env_config['ds']
        self.start_location = env_config['start_location'] if 'start_location' in env_config else 0
        self.travel_distance = env_config['travel_distance']
        self.max_travel_time = self.travel_distance/self.target_v 
        self.ALTI_X = np.array(env_config['ALTI_X'])
        self.ALTI = np.array(env_config['ALTI'])
        self.nelem = math.floor(self.travel_distance/self.ds)+1 #total 
        self.Mveh = env_config['Mveh']
        self.fuel_model = FuelModel(self.Mveh)
        self.loc = [i*self.ds + int(self.start_location) for i in range(self.nelem)]
        self.x0 = [self.v0 for _ in range(self.nelem)]
        self.speed_constraints = env_config['speed_constraints']


    def compute_hx(self, i, x, xi, result):
        h = self.ALTI[i-1] + (self.ALTI[i]-self.ALTI[i-1])*(x-self.ALTI_X[i-1])/(self.ALTI_X[i]-self.ALTI_X[i-1])
        result = result.at[xi].set(h)

        return result

    def do_nothing_compute_hx(self, i, x, xi, result):
        return result

    def query_hnode(self, xnode):
        result = np.array([0.0 for _ in range(len(xnode))])
        for xi in range(len(xnode)):
            for i in range(1,len(self.ALTI_X)):
                result = cond(
                    (self.ALTI_X[i]>xnode[xi]) * (xnode[xi]>=self.ALTI_X[i-1]),
                    self.compute_hx,
                    self.do_nothing_compute_hx,
                    i, 
                    xnode[xi], 
                    xi,
                    result
                )
        return result

    def objective(self, x):

        a =np.array([(x[i+1]**2 - x[i]**2)/(2*self.ds) for i in range(0, len(x)-1)])
        dv = np.array([x[i+1] - x[i] for i in range(0, len(x)-1)])
        a = np.append(a, np.array([0.0]), 0)
        x_node = self.loc
        h_node = self.query_hnode(self.loc)
        # jax.debug.print("loc:{}", self.loc)
        # jax.debug.print("hnode:{}", h_node)
        dx_elem = np.array([x_node[i+1] - x_node[i] for i in range(0, len(x_node)-1)])
        dh_elem = np.array([h_node[i+1] - h_node[i] for i in range(0, len(h_node)-1)])
        Ge = np.arctan(dh_elem/dx_elem)
        Ge = np.append(Ge, np.array([0.0]), 0)
        fuel = self.fuel_model.cal_fuel(a, x, Ge, self.ds)
        # jax.debug.print("fc:{}", fuel)
        # jax.debug.print("ge:{}", Ge)
        # jax.debug.print("a:{}", a)
        total_fuel = 0.0
        for f in fuel:
            total_fuel += f
        
        a_pun = np.sum(np.power(dv, 2))
        # jax.debug.print("apun:{}", a_pun)
        return total_fuel + 10*a_pun

    def eq_constraints(self, x):
        # x = 0
        return x[0]-self.v0

    def ineq_constrains(self, x):
        # x >=0
        total_t = 0
        total_d = 0
        for i in range(len(x)-1):
            total_t += 2*self.ds/(x[i]+x[i+1])
            total_d += self.ds
        # jax.debug.print("total_t:{}", total_t)
        return self.max_travel_time  - total_t


    def solve(self):
        # jit the functions
        obj_jit = jit(self.objective)
        con_eq_jit = jit(self.eq_constraints)
        con_ineq_jit = jit(self.ineq_constrains)

        # build the derivatives and jit them
        obj_grad = jit(grad(obj_jit))  # objective gradient
        obj_hess = jit(jacrev(jacfwd(obj_jit))) # objective hessian
        con_eq_jac = jit(jacfwd(con_eq_jit))  # jacobian
        con_ineq_jac = jit(jacfwd(con_ineq_jit))  # jacobian
        con_eq_hess = jacrev(jacfwd(con_eq_jit)) # hessian
        con_eq_hessvp = jit(lambda x, v: con_eq_hess(x) * v[0]) # hessian vector-product
        con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
        con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0]) # hessian vector-product


        # constraints
        cons = [
            {'type': 'eq', 'fun': con_eq_jit, 'jac': con_eq_jac, 'hess': con_eq_hessvp},
            {'type': 'ineq', 'fun': con_ineq_jit, 'jac': con_ineq_jac, 'hess': con_ineq_hessvp}
        ]


        # variable bounds: 1 <= x[i] <= 5
        bnds = [(self.speed_constraints[0], self.speed_constraints[1]) for _ in range(len(self.x0))]

        # executing the solver
        res = minimize_ipopt(obj_jit, jac=obj_grad, hess=obj_hess, x0=self.x0, bounds=bnds,
                        constraints=cons, options={'print_level': 5})

        return res

if __name__ == "__main__":
    env_config = {
        'ALTI_X': [0, 2000, 3000, 4000, 5000, 8000, 10000, 1e6],
        'ALTI': [0,    20,   30,   40,  20,  -10,   0,   0],
        'Mveh': 55e3,
        'target_v': 72/3.6,
        'ds': 100,
        'start_location': 0,
        'travel_distance': 10000,
        'obs_horizon':2000,
        'obs_step':20,
        'acc_constraints':[-1,1],
        'speed_constraints':[1.0, 50.0]
    }

    pcc = PCC_alg(env_config)
    res = pcc.solve()
    print(res)