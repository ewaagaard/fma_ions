"""
Container for helper functions during tracking to calculate beam parameters and store data
"""
import numpy as np
import xpart as xp
import xtrack as xt
from dataclasses import dataclass
from typing import Self
import pandas as pd

####### Helper functions for bunch length, momentum spread and geometric emittances #######
def _bunch_length(parts: xp.Particles) -> float:
    return np.std(parts.zeta[parts.state > 0])


def _sigma_delta(parts: xp.Particles) -> float:
    return np.std(parts.delta[parts.state > 0])


def _geom_epsx(parts: xp.Particles, twiss: xt.TwissTable) -> float:
    """
    We index dx and betx at 0 which corresponds to the beginning / end of
    the line, since this is where / when we will be applying the kicks.
    """
    sigma_x = np.std(parts.x[parts.state > 0])
    sig_delta = _sigma_delta(parts)
    return (sigma_x**2 - (twiss["dx"][0] * sig_delta) ** 2) / twiss["betx"][0]


def _geom_epsy(parts: xp.Particles, twiss: xt.TwissTable) -> float:
    """
    We index dy and bety at 0 which corresponds to the beginning / end of
    the line, since this is where / when we will be applying the kicks.
    """
    sigma_y = np.std(parts.y[parts.state > 0])
    sig_delta = _sigma_delta(parts)
    return (sigma_y**2 - (twiss["dy"][0] * sig_delta) ** 2) / twiss["bety"][0]
#############################################################################


@dataclass
class Records:
    """
    Data class to store numpy.ndarray of results during tracking 
    - here NORMALIZED emittance is used
    """
    nepsilon_x: np.ndarray
    nepsilon_y: np.ndarray
    sigma_delta: np.ndarray
    bunch_length: np.ndarray
    Nb: np.ndarray

    def update_at_turn(self, turn: int, parts: xp.Particles, twiss: xt.TwissTable):
        """Automatically update the records at given turn from the xpart.Particles."""

        # Store the particle ensemble quantities
        self.nepsilon_x[turn] = _geom_epsx(parts, twiss) * parts.beta0[0] * parts.gamma0[0]
        self.nepsilon_y[turn] = _geom_epsy(parts, twiss) * parts.beta0[0] * parts.gamma0[0]
        self.Nb[turn] = parts.weight[parts.state > 0][0]*len(parts.x[parts.state > 0])
        self.sigma_delta[turn] = _sigma_delta(parts)
        self.bunch_length[turn] = _bunch_length(parts)

    @classmethod
    def init_zeroes(cls, n_turns: int) -> Self:  # noqa: F821
        """Initialize the dataclass with arrays of zeroes."""
        return cls(
            nepsilon_x=np.zeros(n_turns, dtype=float),
            nepsilon_y=np.zeros(n_turns, dtype=float),
            Nb=np.zeros(n_turns, dtype=float),
            sigma_delta=np.zeros(n_turns, dtype=float),
            bunch_length=np.zeros(n_turns, dtype=float)
        )
    
    def to_dict(self):
        return {
            'exn': self.nepsilon_x,
            'eyn': self.nepsilon_y,
            'sigma_delta': self.sigma_delta,
            'bunch_length': self.bunch_length,
            'Nb' : self.Nb
        }
    
# Set up a dataclass to store the results - also growth rates 
@dataclass
class Records_Growth_Rates:
    """
    Data class to store numpy.ndarray of results during tracking 
    Here GEOMETRIC emittance is used to facilitate growth rate calculation
    """
    epsilon_x: np.ndarray
    epsilon_y: np.ndarray
    sigma_delta: np.ndarray
    bunch_length: np.ndarray
    Nb: np.ndarray
    Tx: np.ndarray
    Ty: np.ndarray
    Tz: np.ndarray

    def update_at_turn(self, turn: int, parts: xp.Particles, twiss: xt.TwissTable):
        self.epsilon_x[turn] = _geom_epsx(parts, twiss)
        self.epsilon_y[turn] = _geom_epsy(parts, twiss)
        self.Nb[turn] = parts.weight[parts.state > 0][0]*len(parts.x[parts.state > 0])
        self.sigma_delta[turn] = _sigma_delta(parts)
        self.bunch_length[turn] = _bunch_length(parts)

    @classmethod
    def init_zeroes(cls, n_turns: int) -> Self:  # noqa: F821
        return cls(
            epsilon_x=np.zeros(n_turns, dtype=float),
            epsilon_y=np.zeros(n_turns, dtype=float),
            sigma_delta=np.zeros(n_turns, dtype=float),
            bunch_length=np.zeros(n_turns, dtype=float),
            Nb=np.zeros(n_turns, dtype=float),
            Tx=np.zeros(n_turns, dtype=float),
            Ty=np.zeros(n_turns, dtype=float),
            Tz=np.zeros(n_turns, dtype=float)
        )
    
    def to_dict(self):
        return {
            'ex': self.epsilon_x,
            'ey': self.epsilon_y,
            'sigma_delta': self.sigma_delta,
            'bunch_length': self.bunch_length,
            'Nb' : self.Nb,
            'Tx' : self.Tx,
            'Ty' : self.Ty,
            'Tz' : self.Tz
        }
    
    def to_pandas(self):
        df = pd.DataFrame(self.to_dict())
        return df
        
    
@dataclass
class Full_Records:
    """
    Data class to store numpy.ndarray of results during tracking - also individual particle positions and state
    """
    x: np.ndarray
    y: np.ndarray
    nepsilon_x: np.ndarray
    nepsilon_y: np.ndarray
    sigma_delta: np.ndarray
    bunch_length: np.ndarray
    Nb: np.ndarray
    state: np.ndarray

    def update_at_turn(self, turn: int, parts: xp.Particles, twiss: xt.TwissTable, which_context='cpu'):
        """Automatically update the records at given turn from the xpart.Particles."""

        # Store the particle ensemble quantities
        self.nepsilon_x[turn] = _geom_epsx(parts, twiss) * parts.beta0[0] * parts.gamma0[0]
        self.nepsilon_y[turn] = _geom_epsy(parts, twiss) * parts.beta0[0] * parts.gamma0[0]
        self.Nb[turn] = parts.weight[0]*len(parts.x[parts.state > 0])
        self.sigma_delta[turn] = _sigma_delta(parts)
        self.bunch_length[turn] = _bunch_length(parts)

        # Depending on context, save individual particle values
        if which_context=='cpu':
            self.x[:, turn] = parts.x[parts.state > 0]
            self.y[:, turn] = parts.y[parts.state > 0]
            self.state[:, turn] = parts.state
        elif which_context=='gpu':
            self.x[:, turn] = parts.x[parts.state > 0].get()
            self.y[:, turn] = parts.y[parts.state > 0].get()
            self.state[:, turn] = parts.state.get()

    @classmethod
    def init_zeroes(cls, num_part : int, n_turns: int) -> Self:  # noqa: F821
        """Initialize the dataclass with arrays of zeroes."""
        return cls(
            x=np.zeros([num_part, n_turns], dtype=float),
            y=np.zeros([num_part, n_turns], dtype=float),
            state=np.zeros([num_part, n_turns], dtype=float),
            nepsilon_x=np.zeros(n_turns, dtype=float),
            nepsilon_y=np.zeros(n_turns, dtype=float),
            Nb=np.zeros(n_turns, dtype=float),
            sigma_delta=np.zeros(n_turns, dtype=float),
            bunch_length=np.zeros(n_turns, dtype=float)
        )