"""Comparison of optimiser functions to optimise electric field
"""

import pickle
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
import drop


def study_optimiser(flow:drop.ChannelFlow, t_step:float, e_max:float, num_phase=1,
        opt_method='nelder-mead'):
    """Same function as drop.optimise_e_field(), but with new parameter opt_method
    to set different methods for minimise function
    """

    init_droplets = flow.droplets
    init_conds = [flow.e_field.amplitude, flow.e_field.freq, flow.e_field.bias]

    init_conds = [flow.e_field.amplitude, flow.e_field.freq, flow.e_field.bias]

    res = minimize(drop.optimising_function, init_conds, method=opt_method,
        args=(flow,t_step,num_phase,e_max,init_droplets))

    return res.x, res.fun


# Flow params
HEIGHT, LENGTH = 0.1, 0.3
U_BULK, U_LIQUID = 20, 12.9

N_DROPS, RANDOM_THETAS = 10, False
P_RATIO = 15

# Electric field params
MAX_FIELD_STRENGTH = 1e6
PARALLEL_FIELD = False

init_field_params = (5e4,100,-2.5e4)

# Solver params
DELTA_T = 5e-4
PHASES = 2

# All the methods listed in scipy document
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
methods = ['Nelder-Mead','Powell','CG','BFGS','Newton-CG','L-BFGS-B','TNC','COBYLA','dogleg',
    'trust-constr','trust-ncg','trust-krylov','trust-exact']


### Actual Code
results = {}

for method in methods:
    print(f"Optimiser method: {method}")

    # flowchannel = drop.ChannelFlow.from_flight(HEIGHT,LENGTH,U_BULK,P_RATIO)
    flowchannel = drop.ChannelFlow(HEIGHT,LENGTH,U_BULK)

    # Create droplets
    droplet_diams = drop.gen_droplet_sizes(N_DROPS)
    spray_thetas = np.zeros(N_DROPS)

    for diam,theta in zip(droplet_diams, spray_thetas):
        u_i = U_LIQUID * np.sin(theta)
        v_i = U_LIQUID * np.cos(theta)
        flowchannel.add_droplet(drop.Droplet(diam), u_init=u_i, v_init=v_i)

    # Establish baseline score
    flowchannel.assign_electric_field(drop.ElectricField(0,0,0,parallel=PARALLEL_FIELD))
    baseline_score = drop.optimising_function([0,0,0], flow=flowchannel, t_step=DELTA_T,
        num_phase=PHASES, e_max=MAX_FIELD_STRENGTH, drops=flowchannel.droplets)

    # Find best electric field parameters
    e_field = drop.ElectricField(*init_field_params, parallel=PARALLEL_FIELD)
    flowchannel.assign_electric_field(e_field)

    try:
        start_time = perf_counter()
        e_field_params, opt_score = study_optimiser(flowchannel, DELTA_T,
            e_max=MAX_FIELD_STRENGTH, num_phase=PHASES, opt_method=method)
        end_time = perf_counter()

        results[method] = (e_field_params,[opt_score, baseline_score],end_time-start_time)

    except ValueError as err: # if Jacobian is required
        results[method] = err

# Save results
with open("optimiser_study.pkl","wb") as f:
    pickle.dump(results,f)

print(results)
