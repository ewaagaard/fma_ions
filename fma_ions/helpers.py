"""
Container for helper functions during tracking to calculate beam parameters and store data
"""
import numpy as np
import xpart as xp
import xtrack as xt
from dataclasses import dataclass
from typing import Self
import pandas as pd
import json

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
    px: np.ndarray
    py: np.ndarray
    delta: np.ndarray
    zeta: np.ndarray
    nepsilon_x: np.ndarray
    nepsilon_y: np.ndarray
    sigma_delta: np.ndarray
    bunch_length: np.ndarray
    Nb: np.ndarray
    state: np.ndarray
    which_context : str
    full_data_turn_ind: np.ndarray

    def update_at_turn(self, turn: int, parts: xp.Particles, twiss: xt.TwissTable):
        """Automatically update the records at given turn from the xpart.Particles."""

        # Store the particle ensemble quantities
        self.nepsilon_x[turn] = _geom_epsx(parts, twiss) * parts.beta0[0] * parts.gamma0[0]
        self.nepsilon_y[turn] = _geom_epsy(parts, twiss) * parts.beta0[0] * parts.gamma0[0]
        self.Nb[turn] = parts.weight[0]*len(parts.x[parts.state > 0])
        self.sigma_delta[turn] = _sigma_delta(parts)
        self.bunch_length[turn] = _bunch_length(parts)

        # Depending on context, save individual particle values
        if self.which_context=='cpu':
            self.x[:, turn] = parts.x
            self.y[:, turn] = parts.y
            self.px[:, turn] = parts.px
            self.py[:, turn] = parts.py
            self.delta[:, turn] = parts.delta
            self.zeta[:, turn] = parts.zeta
            self.state[:, turn] = parts.state
        elif self.which_context=='gpu':
            self.x[:, turn] = parts.x.get()
            self.y[:, turn] = parts.y.get()
            self.px[:, turn] = parts.px.get()
            self.py[:, turn] = parts.py.get()
            self.delta[:, turn] = parts.delta.get()
            self.zeta[:, turn] = parts.zeta.get()
            self.state[:, turn] = parts.state.get()

    @classmethod
    def init_zeroes(cls, num_part : int, n_turns: int, which_context : str,
                    full_data_turn_ind : np.ndarray) -> Self:  # noqa: F821
        """Initialize the dataclass with arrays of zeroes, also with full_data_ind at which turns data is saved"""
        return cls(
            x=np.zeros([num_part, n_turns], dtype=float),
            y=np.zeros([num_part, n_turns], dtype=float),
            px=np.zeros([num_part, n_turns], dtype=float),
            py=np.zeros([num_part, n_turns], dtype=float),
            delta=np.zeros([num_part, n_turns], dtype=float),
            zeta=np.zeros([num_part, n_turns], dtype=float),
            state=np.zeros([num_part, n_turns], dtype=float),
            nepsilon_x=np.zeros(n_turns, dtype=float),
            nepsilon_y=np.zeros(n_turns, dtype=float),
            Nb=np.zeros(n_turns, dtype=float),
            sigma_delta=np.zeros(n_turns, dtype=float),
            bunch_length=np.zeros(n_turns, dtype=float),
            which_context=which_context,
            full_data_turn_ind = full_data_turn_ind
        )
    
    def to_dict(self):
        return {
            'exn': self.nepsilon_x,
            'eyn': self.nepsilon_y,
            'sigma_delta': self.sigma_delta,
            'bunch_length': self.bunch_length,
            'Nb' : self.Nb,
            'x' : np.array(self.x),
            'px' : np.array(self.px),
            'y' : np.array(self.y),
            'py' : np.array(self.py),
            'delta' : np.array(self.delta),
            'zeta' : np.array(self.zeta),
            'state' : np.array(self.state)
        }
    
    def copy_first_and_last_turns(self):
        """
        Create a copy of the dataclass with only the data for the very first and very last turn.
        """
        return Full_Records(
            x=np.hstack((self.x[:, :1], self.x[:, -1:])),
            y=np.hstack((self.y[:, :1], self.y[:, -1:])),
            px=np.hstack((self.px[:, :1], self.px[:, -1:])),
            py=np.hstack((self.py[:, :1], self.py[:, -1:])),
            delta=np.hstack((self.delta[:, :1], self.delta[:, -1:])),
            zeta=np.hstack((self.zeta[:, :1], self.zeta[:, -1:])),
            nepsilon_x=np.hstack((self.nepsilon_x[:1], self.nepsilon_x[-1:])),
            nepsilon_y=np.hstack((self.nepsilon_y[:1], self.nepsilon_y[-1:])),
            sigma_delta=np.hstack((self.sigma_delta[:1], self.sigma_delta[-1:])),
            bunch_length=np.hstack((self.bunch_length[:1], self.bunch_length[-1:])),
            Nb=np.hstack((self.Nb[:1], self.Nb[-1:])),
            state=np.hstack((self.state[:, :1], self.state[:, -1:])),
            which_context=self.which_context
        )

    def to_json(self, file_path):
        """
        Save the data to a JSON file.
        """
        data = {
            'x': self.x.tolist(),
            'y': self.y.tolist(),
            'px': self.px.tolist(),
            'py': self.py.tolist(),
            'delta': self.delta.tolist(),
            'zeta': self.zeta.tolist(),
            'nepsilon_x': self.nepsilon_x.tolist(),
            'nepsilon_y': self.nepsilon_y.tolist(),
            'sigma_delta': self.sigma_delta.tolist(),
            'bunch_length': self.bunch_length.tolist(),
            'Nb': self.Nb.tolist(),
            'state': self.state.tolist(),
            'which_context': self.which_context,
            'full_data_turn_ind': self.full_data_turn_ind.tolist()
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)


    @classmethod
    def from_json(cls, file_path):
        """
        Load the data from a JSON file and construct a Full_Records instance.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        return cls(
            x=np.array(data['x']),
            y=np.array(data['y']),
            px=np.array(data['px']),
            py=np.array(data['py']),
            delta=np.array(data['delta']),
            zeta=np.array(data['zeta']),
            nepsilon_x=np.array(data['nepsilon_x']),
            nepsilon_y=np.array(data['nepsilon_y']),
            sigma_delta=np.array(data['sigma_delta']),
            bunch_length=np.array(data['bunch_length']),
            Nb=np.array(data['Nb']),
            state=np.array(data['state']),
            which_context=data['which_context'],
            full_data_turn_ind=np.array(data['full_data_turn_ind'])
        )