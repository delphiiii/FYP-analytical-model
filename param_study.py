"""Param study of simulation
"""

import pickle
import numpy as np
import drop


### Setting up params
# Flow params
HEIGHT, LENGTH = 0.1, 0.3
U_BULK, U_LIQUID = 10, 12.9

N_DROPS, RANDOM_THETAS = 10, False
P_RATIO = 25

# Electric field params
MAX_FIELD_STRENGTH = 1e6
PARALLEL_FIELD = False
DO_E_FIELD_OPTIMISATION = True

e_field_params = (50000,0,-25000)

# Solver params
DELTA_T = 5e-4
PHASES = 2


### Actual Code
params = {}

for p_ratio in range(5,41):
    print(f"Pressure ratio: {p_ratio}")

    flowchannel = drop.ChannelFlow.from_flight(HEIGHT,LENGTH,U_BULK,p_ratio)

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
    e_field = drop.ElectricField(*e_field_params, parallel=PARALLEL_FIELD)
    flowchannel.assign_electric_field(e_field)
    e_field_params, opt_score = drop.optimise_e_field(flowchannel, DELTA_T,
        e_max=MAX_FIELD_STRENGTH, num_phase=PHASES)

    params[p_ratio] = (e_field_params,[opt_score, baseline_score])

with open("params2.pkl","wb") as f:
    pickle.dump(params,f)

print(params)
