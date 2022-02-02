"""Class and Function Definitions for Analytical Simulation of Charged Fuel Droplet in Flow
"""

from copy import deepcopy

# Math packages
import numpy as np
from scipy.stats import weibull_min as weibull
from scipy.optimize import minimize
from scipy.integrate import ode

# Visualisation packages
from matplotlib import pyplot as plt
import seaborn as sns

# Thermodynamic packages
from fluids import ATMOSPHERE_1976 as Atmosphere
import cantera as ct


class Droplet:
    """Creates a charged spherical fuel droplet

    Attributes
    ----------
        diameter, volume, mass, density, charge_density, total_charge : float
            Property of droplet fluid (SI units)
        x, y : float
            Position of fluid in flow (m)
        u, v : float
            Absolute velocity of fluid (m/s)
        Re_drop : float
            Reynold's number of droplet
        Cd : float
            Drag Coefficient of droplet
        Fd : float
            Drag force on droplet (N)
        Fe : float
            Electrostatic force on droplet (N)
    """

    def __init__(self, diameter:float, *, density=810., charge_density=-1.) -> None:
        """
        Parameters:
        ----------
            diameter : float
                Diameter of droplet (m). Assume sphere
            density : float
                Density of droplet (kg/m^3). Assume Jet A1
            charge_density : float
                Charge density of droplet (C/m^3). Assume uniform
        """

        self.diameter = diameter
        self.density = density
        self.charge_density = charge_density
        self.volume = (self.diameter**3)*np.pi/6
        self.total_charge = self.charge_density * self.volume
        self.mass = self.volume * self.density
        self.set_init_cond()
        self.Re_drop = None
        self.Cd = None
        self.Fd = None
        self.Fe = None

    def set_init_cond(self, x_init=0., y_init=0., u_init=0., v_init=0.) -> None:
        """Sets initial condition of droplet

        Parameters:
        ----------
            x_init, y_init : float
                Initial positions of particle (m)
            u_init, v_init : float
                Initial velocities of particle (m/s)
        """

        self.x = x_init
        self.y = y_init
        self.u = u_init
        self.v = v_init

    def get_v_rel(self, u_flow:float) -> tuple:
        """Gets relative velocity to bulk flow

        Parameters:
        ----------
            u_flow : float
                Bulk flow velocity (assume 1D) (m/s)

        Returns:
        ----------
            (u_rel, v_rel) : tuple
                Relative velocity to flow (m/s)
        """

        return (u_flow-self.u, -self.v)

    def set_Re_drop(self, rho_flow:float, mu_flow:float, u_flow:float) -> None:
        """Calculates Reynold's number of droplet using bulk flow characteristics

        Parameters:
        ----------
            rho_flow, mu_flow, u_bulk : float
                Characteristics of bulk flow (SI units)
        """

        self.Re_drop = rho_flow * self.diameter * np.linalg.norm(self.get_v_rel(u_flow)) / mu_flow

    def set_Fd(self, rho_flow:float, mu_flow:float, u_flow:float) -> None:
        """Calculates drag force of droplet using bulk flow characteristics

        Parameters:
        ----------
            rho_flow, mu_flow, u_bulk : float
                Characteristics of bulk flow (SI units)
        """

        self.set_Re_drop(rho_flow, mu_flow, u_flow)
        v_rel_mag = np.linalg.norm(self.get_v_rel(u_flow))

        if self.Re_drop > 1000: # drag law yuen & chen, cs&t, vol 21, 537-541
            self.Cd = 0.424
        else:
            self.Cd = 24/self.Re_drop*(1+(self.Re_drop**(2/3))/6)

        self.Fd = self.Cd*0.125*rho_flow*(v_rel_mag**2)*np.pi*(self.diameter**2)

    def set_Fe(self, field_strength:float) -> None:
        """Calculates electrostatic force on droplet

        Parameters:
        ----------
            field_strength : float
                Electrostatic field strength (V/m)
        """

        self.Fe = field_strength * self.total_charge


class ElectricField:
    """Creates electric field. Assume sine wave with given amplitude, frequency and phase shift.

    Attributes
    ----------
        amplitude : float
            Amplitude of electric field (V)
        freq : float
            Frequency of electric field (Hz)
        phase_shift : float
            Phase shift of electric field (rad)
        bias : float
            Bias of electric field (V). Only applicable if freq!=0
        parallel_to_flow : bool
            Direction of electric field. True if in parallel to bulk flow
        separation : float
            Separation between top and bottom electrode in flow field (m)

    """

    def __init__(self, amplitude, freq=0., bias=0., *, phase_shift=0., parallel=False) -> None:
        """
        Parameters:
        ----------
            amplitude : float
                Amplitude of electric field (V)
            freq : float
                Frequency of electric field (Hz)
            phase_shift : float
                Phase shift (from sine wave) of electric field (rad)
            bias : float
                Bias of electric field (V). Only applicable if freq!=0
            parallel : bool
                Direction of electric field. True if in parallel to bulk flow
        """

        self.amplitude = amplitude
        self.freq = freq
        self.phase_shift = phase_shift
        self.bias = bias
        self.separation = None
        self.parallel_to_flow = parallel

    def get_field_strength(self, time:float) -> float:
        """Gets electric field strength at given time. Only available after assigning flow field

        Parameters:
        ----------
            time : float
                Time of calculation

        Returns:
        ----------
            field_strength : float
               Electric field strength (V/m)
        """

        if self.separation:
            if self.freq == 0:
                field_strength = self.bias / self.separation
            else:
                volts = self.amplitude * np.sin(2*np.pi*self.freq*time+self.phase_shift) + self.bias
                field_strength =  volts / self.separation
        else:
            print("Separation not set!")
            return None

        return field_strength


class ChannelFlow:
    """Creates flow field for problem. Assumes rectangular channel flow.

    Attributes
    ----------
        height, length : float
            Physical dimensions of flow channel (m)
        u_bulk_0 : float
            Bulk flow velocity of flow in channel (m/s)
        P, T, rho, mu : float
            Physical properties of flow fluid (SI units)
        conductivity, permittivity : float
            Electrical properties of flow fluid (SI units)
        e_field : ElectricField
            Electric field applied to the channel
        droplets : [Droplet]
            List of droplets in the flow

    """

    def __init__(self, height:float, length:float, u_bulk_0:float, *, T=273.15, P=101325,
        conductivity=5.5e-15, permittivity=8.854e-12) -> None:
        """
        Parameters:
        ----------
            height, length : float
                Physical dimensions of channel flow (m)
            u_bulk_0 : float
                Bulk flow velocity of flow in channel (m/s)
            T : float
                Temperature of flow (Kelvins). Assumes STP
            P : float
                Pressure of flow (Pa). Assumes STP
            conductivity, permittivity : float
                Physical properties of flow fluid (SI units). Assumes air at STP
        """

        self.height = height
        self.length = length
        self.u_bulk_0 = u_bulk_0
        self.T = T
        self.P = P
        self.rho = Atmosphere.density(T,P)
        self.mu = Atmosphere.viscosity(T)
        self.conductivity = conductivity
        self.permittivity = permittivity
        self.e_field = None
        self.droplets = []

    @classmethod
    def from_flight(cls, height:float, length:float, u_bulk_0:float, p_ratio:float, *,
        altitude=35000, mach=0.85, isen_eff=0.9):
        """Creates ChannelFlow object from in-flight conditions

        Parameters:
        ---------
            height, length : float
                Physical dimensions of channel flow (m)
            u_bulk_0 : float
                Bulk flow velocity of flow in channel (m/s)
            altitude, mach : float
                Flight conditions (ft, mach number)
            p_ratio, isen_eff : float
                Compressor properties

        Returns:
        ----------
            flow : ChannelFlow
                Flow object at given conditions
        """

        atm = Atmosphere(altitude*0.3048) # convert ft to m
        air = ct.Solution('air.yaml')
        air.basis = 'mass'
        air.TP = atm.T, atm.P
        gamma = air.cp/air.cv
        T_02 = air.T * (1+(mach**2)*(gamma-1)/2)
        P_02 = air.P * (1+(mach**2)*(gamma-1)/2)**(gamma/(gamma-1))
        air.TP = T_02, P_02

        h_02, s_02 = air.h, air.s
        air.SP = s_02, P_02*p_ratio
        h_03s = air.h
        h_03a = h_02 + (h_03s-h_02)/isen_eff
        air.HP = h_03a, P_02*p_ratio

        return cls(height,length,u_bulk_0,T=air.T,P=air.P)

    def get_bulk_vel(self, y:float) -> float:
        """Returns turbulant flow velocity at given y (m/s)

        Parameters:
        ----------
            y : float
                Height at which velocity is calculated (m)

        Returns:
        ----------
            u : float
                Flow velocity at given y (m/s)
        """

        return self.u_bulk_0/0.8*((abs(1-abs(2*y)/self.height))**(1/4))

    def assign_electric_field(self, field:ElectricField) -> None:
        """Assigns electric field to flow channel.
        Also calculates field strength based on field and flow properties

        Parameters:
        ---------
            field : Electric_Field
                Electric field to be assigned
        """

        if field.parallel_to_flow is False:
            field.separation = self.height
        else:
            field.separation = self.length

        self.e_field = field

    def add_droplet(self, droplet:Droplet, x_init=0.05, y_init=None, u_init=0., v_init=1.) -> None:
        """Adds droplet to flow and sets initial conditions

        Parameters:
        ---------
            droplet : Droplet
                Droplet to be added into flow
            x_init, y_init : float
                Initial positions of particle (m). Defaults (0.1, -(height/2))
            u_init, v_init : float
                Initial velocities of particle (m/s). Defaults (0, 1)
        """

        y_init = -(self.height/2) if y_init is None else y_init

        droplet.set_init_cond(x_init, y_init, u_init, v_init)
        self.droplets.append(droplet)

    def set_Fd(self) -> None:
        """Calculates drag force on all droplets using flow profile
        """

        for droplet in self.droplets:
            droplet.set_Fd(self.rho, self.mu, self.get_bulk_vel(droplet.y))

    def set_Fe(self, t=0.) -> None:
        """Calculates electrostatic force on all droplets
        Sets Fe to 0 if no field is assigned

        Parameters:
        ---------
            time : float
                Time of calculation for variable field (s)
        """

        if self.e_field:
            for droplet in self.droplets:
                droplet.set_Fe(self.e_field.get_field_strength(t))
        else:
            for droplet in self.droplets:
                droplet.set_Fe(0)


def gen_random_thetas(size:int, *, nozzle_range=20.):
    """Generates random droplet spray angles using Normal distribution
    Data taken from Ahmed (Fuel, 2021)

    Parameters:
    ----------
        size : int
            Number of samples required
        nozzle_range : float
            Estimated 3 sigma range of droplets

    Returns:
    ----------
        thetas : list
            Exit angles of droplets in radians with len(thetas)=size
    """

    thetas = np.random.normal(scale=nozzle_range/3,size=size)
    return thetas*np.pi/180

def gen_droplet_sizes(size:int, *, random_sizes=False):
    """Generates droplet diameters using Weibull (Rosin-Rammler) distribution
    Data taken from Ahmed (Fuel, 2021)

    Parameters:
    ----------
        size : int
            Number of samples required
        random_sizes : bool
            Random sizes or evenly spaced sizes

    Returns:
    ----------
        diams : list
            Diameters of droplets with len(diams)=size
    """

    shape = 1.35 # shape
    dist = weibull(shape)
    if random_sizes is False:
        drop_sizes = dist.ppf(np.linspace(0.05, 0.95, size+2))
    else:
        drop_sizes = dist.rvs(size)

    return drop_sizes[1:-1]*1e-4

def solve(flow:ChannelFlow, t_step:float, num_phase=1):
    """Solves the input flow problem at given times

    Parameters:
    ----------
        flow : ChannelFlow
            Fully defined flow object (with corresponding droplets) wish to be solved
        t_step : float
            Time step for solver (s)
        num_phase : int
            Number of evenly spaced phases to solve together

    Returns:
    ----------
        sols : list
            Solutions at each time step ([x,y,u,v]) for each droplet in flow parameter
        (es_approx_param, mag_neglig_param) : (float, float)
            Validity factors
    """

    def _ode_system(curr_time, curr_state, flow:ChannelFlow, drop:Droplet):
        """Parametric function for ODE solver

        Parameters (passed with set_f_params()):
        ----------
            flow : ChannelFlow
                Flow object to be solved
            drop : Droplet
                Droplet object to be solved

        Returns:
        ----------
            sols : list
                Solution at current time and state ([x,y,u,v])
        """

        drop.x, drop.y, drop.u, drop.v = curr_state # get current state

        u_b = flow.get_bulk_vel(drop.y)
        v_rel = drop.get_v_rel(u_b)
        v_hat = v_rel / np.linalg.norm(v_rel) # get magnitude and direction of v_rel

        flow.set_Fd() # calculate drag force
        flow.set_Fe(curr_time) # calculate electric force

        if flow.e_field.parallel_to_flow is False:
            x_dot, y_dot = drop.u, drop.v
            x_dot_dot = (drop.Fd * v_hat[0]) / drop.mass # a = F/m
            y_dot_dot = (drop.Fd * v_hat[1] - drop.Fe) / drop.mass # a = F/m
        else:
            x_dot, y_dot = drop.u, drop.v
            x_dot_dot = (drop.Fd * v_hat[0] - drop.Fe) / drop.mass # a = F/m
            y_dot_dot = (drop.Fd * v_hat[1]) / drop.mass # a = F/m

        return [x_dot, y_dot, x_dot_dot, y_dot_dot]

    # electrostatic assumption parameters (needs to be <<1) (Zhakin 2012 Phys.-Usp. 55 465)
    es_approx_param = flow.permittivity * flow.e_field.freq * flow.height / (2.99e8)
    mag_neglig_param = flow.conductivity * flow.height / (flow.permittivity * 2.99e8)

    sols = []
    for droplet in flow.droplets: # solve for each droplet
        phase_sol = {}
        initial = [droplet.x, droplet.y, droplet.u, droplet.v]
        for phase in np.linspace(0,2*np.pi,num_phase,endpoint=False): #solve for each phase
            flow.e_field.phase_shift = phase
            solver = ode(_ode_system)
            solver.set_integrator('lsoda')
            solver.set_initial_value(initial)
            solver.set_f_params(flow, droplet)
            sol = np.array(initial)
            while (solver.y[0]<1.1*flow.length # does not reache end of channel
                    and solver.successful()):
                sol = np.vstack((sol, solver.integrate(solver.t+t_step)))
            phase_sol[phase] = sol
        sols.append(phase_sol)

    return sols, (es_approx_param, mag_neglig_param)

def optimise_e_field(flow:ChannelFlow, t_step:float, *, e_max:float, num_phase=1):
    """Optimises for best electric field parameters with given flow problem

    Parameters:
    ----------
        flow : ChannelFlow
            Fully defined flow object (with electric field) wishing to be optimised
        t_step : float
            Time step for optimiser (s)
        e_max : float
            Maximum allowable electric field strength (V/m)
        num_phase : int
            Number of evenly spaced phases to solve together

    Returns:
    ----------
        opt_conditions : list
            Optimal electric field conditions ([amplitude, frequency, bias])
        final_score : float
            final score of field
    """

    def _optimising_function(e_field_props, flow:ChannelFlow, t_step:float,
        num_phase:int, e_max:float, drops:list[Droplet]):
        """Optimises electric field parameters for given flow
        Each droplet is assigned a penalty score based on closeness to wall
        Aims to minimise the penalty score

        Parameters (passed with args='' argument):
        ----------
            local_flow : ChannelFlow
                Fully defined flow object (with corresponding droplets) containing flow conditions
            t_step : float
                Time step for solver (s)
            num_phase : int
                Number of evenly spaced phases to solve together
            e_max : float
                Maximum field strength allowed (V/m)
            drops : [Droplet]
                Initial droplets

        Returns:
        ----------
            score : float
                Score associated with given electric field parameters
        """

        # Create new set of objects
        local_flow = deepcopy(flow)
        local_flow.droplets = deepcopy(drops)

        # Create new electric field based on new conditions
        local_e_field = ElectricField(*e_field_props,parallel=local_flow.e_field.parallel_to_flow)
        local_flow.assign_electric_field(local_e_field)

        # Solve new problem
        local_solutions, _ = solve(local_flow, t_step=t_step, num_phase=num_phase)

        thresh = round(4e-3/t_step) # to ignore initial upwards trajetory
        score = 0

        for local_solution in local_solutions: # process each droplet separately
            sol_array = np.array([])
            for local_phase_solution in local_solution.values(): # append y values for all phases
                sol_array = np.append(sol_array, local_phase_solution[thresh:,1])

            dist_1 = max(sol_array)/(local_flow.height/2) # non-dimensionalised distance to top wall
            dist_2 = min(sol_array)/(local_flow.height/2) # non-dimensionalised distance to lower wall
            dist = max(abs(dist_1),abs(dist_2)) # use maximum for penalty
            score += dist

        # Additional penalty if field strength is too high (i.e. risk of breakdown)
        e_strength_max = abs(local_e_field.amplitude)+abs(local_e_field.bias)/local_e_field.separation
        if e_strength_max > e_max:
            score *= (1+e_strength_max-e_max)

        print(f"{e_field_props[0]:.2e}V, {e_field_props[1]:.2f}Hz, {e_field_props[2]:.2e}V bias")
        print(f"Score: {score}")
        return score

    init_droplets = flow.droplets
    init_conds = [flow.e_field.amplitude, flow.e_field.freq, flow.e_field.bias]

    res = minimize(_optimising_function, init_conds, method='Nelder-Mead',
        bounds=((0,1e6),(0,1e4),(-1e6,1e6)), args=(flow,t_step,num_phase,e_max,init_droplets))

    return res.x, res.fun

def visualise(flow_solution, channel_length=0.3, channel_height=0.1, *, labels=None, title=None):
    """Visualises the results of flow solution

    Parameters:
    ----------
        flow_solution : array_like
            Full array of solutions
        channel_length, channel_height : float, float
            Dimensions of flow channel
        labels, title : str, str
            labels and title for plot

    Returns:
    ----------
        fig, ax : Matplotlib objects
            Matplotlib plot of input problem
    """

    fig, axis = plt.subplots(figsize=(channel_length*50, channel_height*50))

    for stream_sol in flow_solution:
        drop_sol = np.array([])
        for phase_sol in stream_sol.values():
            drop_sol = np.append(drop_sol, phase_sol)
        drop_sol = np.reshape(drop_sol, (-1,4))
        sns.lineplot(x=drop_sol[:,0], y=drop_sol[:,1], alpha=0.5, linewidth=3, estimator=None, ax=axis)

    plt.ylim([-channel_height/2, channel_height/2])
    plt.xlim([0, channel_length])
    plt.legend(labels)
    plt.title(title)
    return fig, axis
