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

#### tune ripple helper function ###

def get_k_ripple_summed_signal(n_turns, ripple_periods, kqf_amplitudes, kqd_amplitudes,
                                kqf_phases, kqd_phases):
    """
    Generate noise signal on top of kqf/kqd values, with desired ripple periods and amplitudes.
    Phase and frequencies unit must correspond to where it is used, e.g turns
    
    Parameters:
    -----------
    ripple_periods : np.ndarray
        floats containing the ripple periods of the noise frequencies
    kqf_amplitudes : np.ndarray
        ripple amplitudes for desired frequencies of kqf --> obtained from normalized FFT spectrum of IQD and IQF. 
        Default without 50 Hz compensation is 1e-6
    kqd_amplitudes : list
        ripple amplitudes for desired frequencies of kqd --> obtained from normalized FFT spectrum of IQD and IQF. 
        Default without 50 Hz compensation is 1e-6
    kqf_phases : np.ndarray
        ripple phase for desired frequencies of kqf --> obtained from normalized FFT spectrum of IQD and IQF. 
    kqd_phases : list
        ripple phases for desired frequencies of kqd --> obtained from normalized FFT spectrum of IQD and IQF. 

    Returns:
    --------
    k_ripple_values : np.ndarray
        focusing quadrupole values corresponding to modulate Qx according to dq (if chosen plane)
    """

    turns = np.arange(1, n_turns+1)
    kqf_signals = np.zeros([len(ripple_periods), len(turns)])
    kqd_signals = np.zeros([len(ripple_periods), len(turns)])
    for i, ripple_period in enumerate(ripple_periods):
        kqf_signals[i, :] = kqf_amplitudes[i] * np.sin(2 * np.pi * turns / ripple_period + kqf_phases[i])
        kqd_signals[i, :] = kqd_amplitudes[i] * np.sin(2 * np.pi * turns / ripple_period + kqd_phases[i])

    # Sum the signal
    kqf_ripple = np.sum(kqf_signals, axis=0)
    kqd_ripple = np.sum(kqd_signals, axis=0)

    print('Generated kqf ripple of amplitudes {} and phases {} with ripple periods {}'.format(kqf_amplitudes, kqf_phases, ripple_periods))
    print('Generated kqd ripple of amplitudes {} and phases {} with ripple periods {}'.format(kqd_amplitudes, kqd_phases, ripple_periods))

    return kqf_ripple, kqd_ripple


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
    - normalized emittance is used
    - beam profile data (transverse wire scanner and longitudinal profile monitor) can be added
    - initial and final particle object
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
    includes_particle_and_twiss_data: bool = False
    includes_profile_data: bool = False
    includes_seconds_array: bool = False
    includes_centroid_array: bool = False

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

    def append_profile_monitor_data(self, 
                                    monitorH, 
                                    monitorV, 
                                    seconds_array=None):
        """
        If tracking has been done with installed beam profile monitors, 
        append data to class

        Parameters:
        -----------
        monitorH, monitorV : xt.BeamProfileMonitor
            transverse monitors installed at WS locations
        seconds_array : np.ndarray
            array containing seconds (instead of turns) of tracking
        """

        # Append X and Y WS monitors - convert to lists to save to json        
        self.monitorH_x_grid = monitorH.x_grid.tolist()
        self.monitorH_x_intensity = monitorH.x_intensity.tolist()
        self.monitorV_y_grid = monitorV.y_grid.tolist()
        self.monitorV_y_intensity = monitorV.y_intensity.tolist()

        # Append seconds from tracking 
        if seconds_array is not None:
            self.seconds_array = seconds_array.tolist()

        self.includes_profile_data = True
        self.includes_seconds_array = True if seconds_array is not None else False
        
        
    def append_centroid_data(self, X_data : np.ndarray, Y_data : np.ndarray, 
                             kqf_data : np.ndarray, kqd_data : np.ndarray):
        """
        Append X and Y centroid data and quadrupolar knob from Twiss to object
        """
        self.X_data = X_data.tolist()
        self.Y_data = Y_data.tolist()
        self.kqf_data = kqf_data.tolist()
        self.kqd_data = kqd_data.tolist()
        self.includes_centroid_array = True


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
            'includes_profile_data' : self.includes_profile_data,
            'Turns': self.turns.tolist()
        }
        if self.includes_particle_and_twiss_data:
            data['twiss'] = self.twiss
            data['particles_i'] = self.particles_i
            data['particles_f'] = self.particles_f 
        if self.includes_profile_data:
            data['monitorH_x_grid'] = self.monitorH_x_grid
            data['monitorH_x_intensity'] = self.monitorH_x_intensity
            data['monitorV_y_grid'] = self.monitorV_y_grid
            data['monitorV_y_intensity'] = self.monitorV_y_intensity      
        if self.includes_seconds_array:
            data['Seconds'] = self.seconds_array
        if self.includes_centroid_array:
            data['X_data'] = self.X_data
            data['Y_data'] = self.Y_data
            data['kqf_data'] = self.kqf_data
            data['kqd_data'] = self.kqd_data

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


    @staticmethod
    def dict_from_json(file_path, convert_to_numpy=True):
        """
        Load the data from a JSON file and construct a dictionary from data
        Convert lists in dictionary to numpy format if desired
        """
        with open(file_path, 'r') as f:
            tbt_dict = json.load(f)

        # Convert every list to numpy array
        if convert_to_numpy:
            for key, value in tbt_dict.items():
                if isinstance(tbt_dict[key], list):
                    tbt_dict[key] = np.array(tbt_dict[key])

        return tbt_dict