"""
Container of simplistic record class to store values during tracking
"""
import numpy as np
import xpart as xp
import xtrack as xt
import xobjects as xo
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
    delta = parts.delta[parts.state > 0]
    sigma_x = np.std(parts.x[parts.state > 0])
    sig_delta = _sigma_delta(parts)
    return (sigma_x**2 - (twiss["dx"][0] * sig_delta) ** 2) / twiss["betx"][0]


def _geom_epsy(parts: xp.Particles, twiss: xt.TwissTable) -> float:
    """
    We index dy and bety at 0 which corresponds to the beginning / end of
    the line, since this is where / when we will be applying the kicks.
    """
    delta = parts.delta[parts.state > 0]
    sigma_y =  np.std(parts.y[parts.state > 0])
    sig_delta = _sigma_delta(parts)
    return (sigma_y**2 - (twiss["dy"][0] * sig_delta) ** 2) / twiss["bety"][0]

#############################################################################

@dataclass
class Records:
    """
    Data class to store numpy.ndarray of results during flat-bottom tracking 
    """
    exn: np.ndarray
    eyn: np.ndarray
    sigma_delta: np.ndarray
    bunch_length: np.ndarray
    Nb: np.ndarray
    turns: np.ndarray
    twiss: dict
    particles_i: dict
    particles_f: dict

    def update_at_turn(self, turn: int, parts: xp.Particles, twiss: xt.TwissTable):
        """Automatically update the records at given turn from the xpart.Particles."""

        # Store the particle ensemble quantities
        self.exn[turn] = _geom_epsx(parts, twiss) * parts.beta0[0] * parts.gamma0[0]
        self.eyn[turn] = _geom_epsy(parts, twiss) * parts.beta0[0] * parts.gamma0[0]
        self.Nb[turn] = parts.weight[parts.state > 0][0]*len(parts.x[parts.state > 0])
        self.sigma_delta[turn] = _sigma_delta(parts)
        self.bunch_length[turn] = _bunch_length(parts)

    @classmethod
    def init_zeroes(cls, n_turns: int) -> Self:  # noqa: F821
        """Initialize the dataclass with arrays of zeroes."""
        return cls(
            exn=np.zeros(n_turns, dtype=float),
            eyn=np.zeros(n_turns, dtype=float),
            Nb=np.zeros(n_turns, dtype=float),
            sigma_delta=np.zeros(n_turns, dtype=float),
            bunch_length=np.zeros(n_turns, dtype=float),
            turns=np.arange(n_turns, dtype=int),
            twiss={},
            particles_i={},
            particles_f={}
        )
    
    def store_twiss(self, df_twiss: pd.DataFrame):
        """Store twiss table, before collective elements"""
        self.twiss = df_twiss.to_dict()
        self.includes_particle_and_twiss_data = True

    def store_initial_particles(self, parts: xp.Particles):
        """Store initial particle object"""
        self.particles_i = parts.to_dict()
        self.includes_particle_and_twiss_data = True

    def store_final_particles(self, parts: xp.Particles):
        """Store final particle object"""
        self.particles_f = parts.to_dict()
        self.includes_particle_and_twiss_data = True

    def to_dict(self, convert_to_numpy=True):
        """
        Convert data arrays to dictionary, possible also beam profile monitor data
        Convert lists to numpy format if desired, but typically not if data is saved to json
        """
        data = {
            'exn': self.exn.tolist(),
            'eyn': self.eyn.tolist(),
            'sigma_delta': self.sigma_delta.tolist(),
            'bunch_length': self.bunch_length.tolist(),
            'Nb' : self.Nb.tolist(),
            'Turns': self.turns.tolist()
        }
        data['twiss'] = self.twiss
        data['particles_i'] = self.particles_i
        data['particles_f'] = self.particles_f 

        # Convert lists to numpy arrays if desired
        if convert_to_numpy:
            for key, value in data.items():
                if isinstance(data[key], list):
                    data[key] = np.array(data[key])
                    
        return data


    def to_json(self, file_path=None):
        """
        Save the data to a JSON file.
        """
        if file_path is None:
            file_path = './'

        data = self.to_dict()

        with open('{}tbt.json'.format(file_path), 'w') as f:
            json.dump(data, f, cls=xo.JEncoder)