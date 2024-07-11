"""
Example script to compare aggregated particle quantities over many turns vs TBT data
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import xpart as xp
import xtrack as xt
import xobjects as xo

from typing import Self
from dataclasses import dataclass


def _exn(x: np.ndarray, delta: np.ndarray, state: np.ndarray, twiss: xt.TwissTable) -> float:
    """
    We index dx and betx at 0 which corresponds to the beginning / end of
    the line, since this is where / when we will be applying the kicks.
    """
    sigma_x = np.std(x[state > 0])
    sigma_delta = np.std(delta[state > 0])
    return (sigma_x**2 - (twiss["dx"][0] * sigma_delta) ** 2) / twiss["betx"][0] * twiss.beta0 * twiss.gamma0


def _eyn(y: np.ndarray, delta: np.ndarray, state: np.ndarray, twiss: xt.TwissTable) -> float:
    """
    We index dy and bety at 0 which corresponds to the beginning / end of
    the line, since this is where / when we will be applying the kicks.
    """
    sigma_y = np.std(y[state > 0])
    sigma_delta = np.std(delta[state > 0])
    return (sigma_y**2 - (twiss["dy"][0] * sigma_delta) ** 2) / twiss["bety"][0] * twiss.beta0 * twiss.gamma0


@dataclass
class Records:
    """
    Data class to store numpy.ndarray of results during tracking 
    - normalized emittance is used
    - beam profile data (transverse wire scanner and longitudinal profile monitor) can be added
    """
    exn: np.ndarray
    eyn: np.ndarray
    sigma_delta: np.ndarray
    bunch_length: np.ndarray
    Nb: np.ndarray
    turns: np.ndarray

    def update_at_turn(self, turn: int, parts: xp.Particles, twiss: xt.TwissTable):
        """Automatically update the records at given turn from the xpart.Particles."""

        # Store the particle ensemble quantities
        self.exn[turn] = _exn(parts.x, parts.delta, parts.state, twiss)
        self.eyn[turn] = _eyn(parts.y, parts.delta, parts.state, twiss)
        self.Nb[turn] = parts.weight[parts.state > 0][0]*len(parts.x[parts.state > 0])
        self.sigma_delta[turn] = np.std(parts.delta[parts.state > 0])
        self.bunch_length[turn] = np.std(parts.zeta[parts.state > 0])

    @classmethod
    def init_zeroes(cls, n_turns: int) -> Self:  # noqa: F821
        """Initialize the dataclass with arrays of zeroes."""
        return cls(
            exn=np.zeros(n_turns, dtype=float),
            eyn=np.zeros(n_turns, dtype=float),
            Nb=np.zeros(n_turns, dtype=float),
            sigma_delta=np.zeros(n_turns, dtype=float),
            bunch_length=np.zeros(n_turns, dtype=float),
            turns=np.arange(n_turns, dtype=int)   
        )


@dataclass
class Container:
    """
    Data class to aggregate coordinates over a specified interval
    """
    x: np.ndarray
    y: np.ndarray
    zeta: np.ndarray
    delta: np.ndarray
    state: np.ndarray
    which_context : str = 'cpu'

    def update_at_turn(self, turn: int, parts: xp.Particles):
        """Automatically update the records at given turn from the xpart.Particles."""
        # Depending on context, save individual particle values
        if self.which_context=='cpu':
            self.x[:, turn] = parts.x
            self.y[:, turn] = parts.y
            self.delta[:, turn] = parts.delta
            self.zeta[:, turn] = parts.zeta
            self.state[:, turn] = parts.state
        elif self.which_context=='gpu':
            self.x[:, turn] = parts.x.get()
            self.y[:, turn] = parts.y.get()
            self.delta[:, turn] = parts.delta.get()
            self.zeta[:, turn] = parts.zeta.get()
            self.state[:, turn] = parts.state.get()

    @classmethod
    def init_zeroes(cls, num_part : int, n_turns: int, which_context : str = 'cpu') -> Self:  # noqa: F821
        """Initialize the dataclass with arrays of zeroes, also with full_data_ind at which turns data is saved"""
        return cls(
            x=np.zeros([num_part, n_turns], dtype=float),
            y=np.zeros([num_part, n_turns], dtype=float),
            delta=np.zeros([num_part, n_turns], dtype=float),
            zeta=np.zeros([num_part, n_turns], dtype=float),
            state=np.zeros([num_part, n_turns], dtype=float),
            which_context=which_context,
        )


@dataclass
class Aggregator:
    """
    Data class to stack ensemble quanties from container of aggregated particles every X turn
    """
    exn : np.ndarray
    eyn : np.ndarray
    sigma_delta : np.ndarray
    bunch_length : np.ndarray
    num_turns : int 
    accumulation_interval : int
    index : int = 0 # index to keep track on where to append

    def __post_init__(self):
        self.turn_array = np.arange(accumulation_interval, num_turns + accumulation_interval, step=accumulation_interval)
        
    @classmethod
    def init_monitor(cls, 
                     accumulation_interval : int,
                     num_turns: int) -> Self:  # noqa: F821
        """Initialize dataclass with empty arrays containing ensemble quantities every X turn"""
        n_profiles = int(num_turns / accumulation_interval)
        return cls(
            exn = np.zeros(n_profiles, dtype=float),
            eyn = np.zeros(n_profiles, dtype=float),
            sigma_delta = np.zeros(n_profiles, dtype=float),
            bunch_length = np.zeros(n_profiles, dtype=float),
            num_turns = num_turns,
            accumulation_interval = accumulation_interval
        )

    def get_ensemble_values_from_container(self, container : Container, twiss: xt.TwissTable):
        # Convert data class of particles to histogram
        x = container.x.flatten()
        y = container.y.flatten()
        delta = container.delta.flatten()
        zeta = container.zeta.flatten()
        s = container.state.flatten()

        # Aggregate longitudinal coordinates of particles still alive
        self.exn[self.index] = _exn(x, delta, s, twiss)
        self.eyn[self.index] = _eyn(y, delta, s, twiss)
        self.sigma_delta[self.index] = np.std(delta[s > 0])
        self.bunch_length[self.index] = np.std(zeta[s > 0])
        
        # Move to next step
        self.index += 1


# Parameters for tracking
num_part = 3000
num_turns = 500
accumulation_interval = 100
Nb = 3.5e8
exn = 1.3e-6
eyn = 0.9e-6
sigma_z = 0.21
context = xo.ContextCpu(omp_num_threads='auto')
run_with_aggregator = True

# Load SPS Pb seuqence
line = xt.Line.from_json('SPS_2021_Pb_nominal.json')
twiss = line.twiss()

# Generate matched Gaussian beam
particles = xp.generate_matched_gaussian_bunch(_context=context,
                                                num_particles=num_part, 
                                                total_intensity_particles=Nb,
                                                nemitt_x=exn, 
                                                nemitt_y=eyn, 
                                                sigma_z= sigma_z,
                                                particle_ref=line.particle_ref, 
                                                line=line)


# Initialize a container (to contain particles every 100 turns) and an aggregator
if run_with_aggregator:
    container = Container.init_zeroes(len(particles.x), accumulation_interval, which_context='cpu')
    container.update_at_turn(0, particles)
    aggregator = Aggregator.init_monitor(accumulation_interval, num_turns)

# Initialize turn-by-turn records class
tbt = Records.init_zeroes(num_turns) 
tbt.update_at_turn(0, particles, twiss)

# Start tracking 
time00 = time.time()
for turn in range(1, num_turns):
    if turn % 10 == 0:
        print('Tracking turn {}'.format(turn))   

    # Track particles and fill zeta container
    line.track(particles, num_turns=1)
    tbt.update_at_turn(turn, particles, twiss)
    if run_with_aggregator:
        container.update_at_turn(turn % accumulation_interval, particles) 

    # When time is right, merge container into aggregator
    if ((turn+1) % accumulation_interval == 0) and run_with_aggregator:
                
        # Aggregate longitudinal coordinates of particles still alive - into bunch length
        aggregator.get_ensemble_values_from_container(container, twiss)
        print('Aggregated quanties')

        # Initialize new zeta containers
        del container
        container = Container.init_zeroes(len(particles.x), accumulation_interval)
        container.update_at_turn(0, particles) # start from turn, but 0 in new dataclass
                
time01 = time.time()
dt0 = time01-time00
print('\nTracking time: {:.1f} s = {:.1f} h, with aggregator: {}'.format(dt0, dt0/3600, run_with_aggregator))



# Plot the ensemble quantities
if run_with_aggregator:
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (13, 4.5))
    ax1.plot(tbt.turns, 1e6 * tbt.exn, label='TBT')
    ax1.plot(aggregator.turn_array, 1e6 * aggregator.exn, marker='o', ls='None', label='Aggregated')
    ax1.set_ylabel(r'$\varepsilon_{x}^{n}$ [u rad]')
    
    ax2.plot(tbt.turns, 1e6 * tbt.eyn, label='TBT')
    ax2.plot(aggregator.turn_array, 1e6 * aggregator.eyn, marker='o', ls='None', label='Aggregated')
    ax2.set_ylabel(r'$\varepsilon_{y}^{n}$ [u rad]')
    
    ax3.plot(tbt.turns, tbt.bunch_length, label='TBT')
    ax3.plot(aggregator.turn_array, aggregator.bunch_length, marker='o', ls='None', label='Aggregated')
    ax3.set_ylabel(r'$\sigma_{z}$ [m]')
    
    ax4.plot(tbt.turns, tbt.sigma_delta, label='TBT')
    ax4.plot(aggregator.turn_array, aggregator.sigma_delta, marker='o', ls='None', label='Aggregated')
    ax4.set_ylabel(r'$\sigma_{\delta}$')
    ax4.legend()
    
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlabel('Turns')
    
    f.tight_layout()
    plt.show()