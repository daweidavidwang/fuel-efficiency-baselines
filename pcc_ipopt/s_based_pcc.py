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
v0 = 72/3.6
t_pred = 250
offset = 1000
ds = 50
dist = 5000
time_threshold = dist/v0
nelem = math.floor(dist/ds) #total step

ALTI_X = np.array([0, 2000, 3000, 4000, 5000, 1e9])
ALTI   = np.array([0,    20,   30,   20,    -10,   0])
Mveh = 55e3   # vehicle mass [kg]

def compute_hx(i, x, xi, result):
    h = ALTI[i-1] + (ALTI[i]-ALTI[i-1])*(x-ALTI_X[i-1])/(ALTI_X[i]-ALTI_X[i-1])
    result = result.at[xi].set(h)

    return result

def do_nothing_compute_hx(i, x, xi, result):
    return result

def query_hnode(xnode):
    result = np.array([0.0 for _ in range(len(xnode))])
    for xi in range(len(xnode)):
        for i in range(1,len(ALTI_X)):
            result = cond(
                (ALTI_X[i]>xnode[xi]) * (xnode[xi]>=ALTI_X[i-1]),
                compute_hx,
                do_nothing_compute_hx,
                i, 
                xnode[xi], 
                xi,
                result
            )
    return result
loc = [i*ds for i in range(nelem)]
x0 = [v0 for _ in range(nelem)]
fuel_model = FuelModel(Mveh)

def objective(x):

    a =np.array([(x[i+1]**2 - x[i]**2)/(2*ds) for i in range(0, len(x)-1)])
    a = np.append(a, np.array([0.0]), 0)
    x_node = loc
    h_node = query_hnode(loc)
    # jax.debug.print("loc:{}", loc)
    # jax.debug.print("hnode:{}", h_node)
    dx_elem = np.array([x_node[i+1] - x_node[i] for i in range(0, len(x_node)-1)])
    dh_elem = np.array([h_node[i+1] - h_node[i] for i in range(0, len(h_node)-1)])
    Ge = np.arctan(dh_elem/dx_elem)
    Ge = np.append(Ge, np.array([0.0]), 0)
    fuel = fuel_model.cal_fuel(a, x, Ge, ds)
    # jax.debug.print("fc:{}", fuel)
    # jax.debug.print("ge:{}", Ge)
    # jax.debug.print("a:{}", a)
    total_fuel = 0.0
    for f in fuel:
        total_fuel += f
    
    a_pun = np.sum(np.power(a, 2))
    # jax.debug.print("apun:{}", a_pun)
    return total_fuel + a_pun

def eq_constraints(x):
    # x = 0
    return 0.0

def ineq_constrains(x):
    # x >=0
    total_t = 0
    for i in range(len(x)-1):
        total_t += 2*ds/(x[i]+x[i+1])

    return time_threshold - total_t

# jit the functions
obj_jit = jit(objective)
con_eq_jit = jit(eq_constraints)
con_ineq_jit = jit(ineq_constrains)

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
bnds = [(0, 50) for _ in range(len(x0))]

# executing the solver
res = minimize_ipopt(obj_jit, jac=obj_grad, hess=obj_hess, x0=x0, bounds=bnds,
                  constraints=cons, options={'disp': 5})

print(res)