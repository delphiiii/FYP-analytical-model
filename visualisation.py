"""Modelling and visualisation of results of analytical simulation
"""

import numpy as np
from matplotlib import pyplot as plt
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
# Create flow channel
flowchannel = drop.ChannelFlow.from_flight(HEIGHT,LENGTH,U_BULK,P_RATIO)
# flowchannel = drop.ChannelFlow(HEIGHT,LENGTH,U_BULK)

# Create droplets
droplet_diams = drop.gen_droplet_sizes(N_DROPS)
if RANDOM_THETAS:
    spray_thetas = drop.gen_random_thetas(N_DROPS)
else:
    spray_thetas = np.zeros(N_DROPS)

plot_labels = [] # use later for plotting
for diam,theta in zip(droplet_diams, spray_thetas):
    u_i = U_LIQUID * np.sin(theta)
    v_i = U_LIQUID * np.cos(theta)
    flowchannel.add_droplet(drop.Droplet(diam), u_init=u_i, v_init=v_i)
    plot_labels.append(f"d={diam*1e6:.2f}um, theta={theta*180/np.pi:.2f}deg")

# Create electric field
e_field = drop.ElectricField(*e_field_params, parallel=PARALLEL_FIELD)
flowchannel.assign_electric_field(e_field)
if DO_E_FIELD_OPTIMISATION is True:
    e_field_params, e_field_score = drop.optimise_e_field(flowchannel, DELTA_T,
        e_max=MAX_FIELD_STRENGTH, num_phase=PHASES)
    flowchannel.assign_electric_field(e_field)

print(f"{flowchannel.e_field.amplitude:.2e}V, {flowchannel.e_field.freq:.2f}Hz, \
    {flowchannel.e_field.bias:.2e}V bias.")

title = f"u_bulk={U_BULK:.2f}. u_liquid={U_LIQUID:.2f} \
    field: {e_field_params[0]:.2e}V, {e_field_params[1]:.2f}Hz, {e_field_params[2]:.2e}V bias."

fig, ax = drop.visualise(flowchannel, DELTA_T, n_phase=PHASES, labels=plot_labels, title=title)
plt.title(title)
plt.show()