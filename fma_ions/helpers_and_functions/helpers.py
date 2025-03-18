"""
Container for helper functions during tracking to calculate beam parameters and store data
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
class FMA_keeper:
    """
    Data class to store particle object properties
    """
    x: np.ndarray
    y: np.ndarray
    px: np.ndarray
    py: np.ndarray
    Qx: np.ndarray
    Qy: np.ndarray
    d: np.ndarray
    tune_data_is_available = False

    @classmethod
    def init_zeroes(cls, n_turns: int) -> Self:  # noqa: F821
        """Initialize the dataclass with arrays of zeroes."""
        return cls(
            x=np.zeros(n_turns, dtype=float),
            y=np.zeros(n_turns, dtype=float),
            px=np.zeros(n_turns, dtype=float),
            py=np.zeros(n_turns, dtype=float),
            Qx=np.zeros(n_turns, dtype=float),
            Qy=np.zeros(n_turns, dtype=float),
            d=np.zeros(n_turns, dtype=float),
        )

    def update_at_turn(self, turn: int, parts: xp.Particles, context: xo.context):
        """Automatically update the keeper class"""

        # Store the particle ensemble quantities
        self.x[turn] = context.nparray_from_context_array(parts.x)
        self.y[turn] = context.nparray_from_context_array(parts.y)
        self.px[turn] = context.nparray_from_context_array(parts.px)
        self.py[turn] = context.nparray_from_context_array(parts.py)


    def to_dict(self, convert_to_numpy=True):
        """
        Convert data arrays to dictionary, possible also beam profile monitor data
        Convert lists to numpy format if desired, but typically not if data is saved to json
        """
        data = {
            'x': self.x.tolist(),
            'y': self.y.tolist(),
            'px': self.x.tolist(),
            'py': self.y.tolist(),
            'turns': self.turns.tolist()
        }
        if self.tune_data_is_available:
            data['Qx'] = self.Qx
            data['Qy'] = self.Qy
            data['d'] = self.d
        # Convert lists to numpy arrays if desired
        if convert_to_numpy:
            for key, value in data.items():
                if isinstance(data[key], list):
                    data[key] = np.array(data[key])
                    
        return data


    def add_tune_data_to_dict(self, Qx, Qy, d):
        """
        Add FMA data (tunes Qx, Qy and diffusion coefficient d) to dictionary
        """
        self.Qx = Qx
        self.Qy = Qy
        self.d = d
        self.tune_data_is_available = True


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
                                    monitorZ, 
                                    seconds_array=None,
                                    also_keep_delta_profiles=False):
        """
        If tracking has been done with installed beam profile monitors, 
        append data to class

        Parameters:
        -----------
        monitorH, monitorV : xt.BeamProfileMonitor
            transverse monitors installed at WS locations
        monitorZ : fma_ions.Longitudinal_Monitor
            longitudinal beam monitor
        seconds_array : np.ndarray
            array containing seconds (instead of turns) of tracking
        also_keep_delta_profiles : bool
            whether to keep aggregated delta coordinates in Zeta_Container or not
        """

        # Append X and Y WS monitors - convert to lists to save to json        
        self.monitorH_x_grid = monitorH.x_grid.tolist()
        self.monitorH_x_intensity = monitorH.x_intensity.tolist()
        self.monitorV_y_grid = monitorV.y_grid.tolist()
        self.monitorV_y_intensity = monitorV.y_intensity.tolist()

        ### Append data fron longitudinal monitor
        self.nturns_profile_accumulation_interval = monitorZ.nturns_profile_accumulation_interval
        self.z_bin_centers = monitorZ.z_bin_centers.tolist()
        self.z_bin_heights = monitorZ.z_bin_heights.tolist()

        # Whether to keep delta profiles or not
        if also_keep_delta_profiles:
            self.delta_bin_centers = monitorZ.delta_bin_centers.tolist()
            self.delta_bin_heights = monitorZ.delta_bin_heights.tolist()
        self.also_keep_delta_profiles = also_keep_delta_profiles

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
            data['nturns_profile_accumulation_interval'] = self.nturns_profile_accumulation_interval
            data['z_bin_centers'] = self.z_bin_centers
            data['z_bin_heights'] = self.z_bin_heights
            if self.also_keep_delta_profiles:
                data['delta_bin_centers'] = self.delta_bin_centers
                data['delta_bin_heights'] = self.delta_bin_heights                
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


@dataclass
class Zeta_Container:
    """
    Data class to store longitudinal zeta and delta coordinates
    """
    zeta: np.ndarray
    delta: np.ndarray
    state: np.ndarray
    which_context : str

    def update_at_turn(self, turn: int, parts: xp.Particles):
        """Automatically update the records at given turn from the xpart.Particles."""
        # Depending on context, save individual particle values
        if self.which_context=='cpu':
            self.zeta[:, turn] = parts.zeta
            self.delta[:, turn] = parts.delta
            self.state[:, turn] = parts.state
        elif self.which_context=='gpu':
            self.zeta[:, turn] = parts.zeta.get()
            self.delta[:, turn] = parts.delta.get()
            self.state[:, turn] = parts.state.get()

    @classmethod
    def init_zeroes(cls, num_part : int, n_turns: int, which_context : str) -> Self:  # noqa: F821
        """Initialize the dataclass with arrays of zeroes, also with full_data_ind at which turns data is saved"""
        return cls(
            zeta=np.zeros([num_part, n_turns], dtype=float),
            delta=np.zeros([num_part, n_turns], dtype=float),
            state=np.zeros([num_part, n_turns], dtype=float),
            which_context=which_context,
        )


@dataclass
class Longitudinal_Monitor:
    """
    Data class to store histogram of longitudinal zeta coordinate and delta
    """
    z_bin_centers : np.ndarray
    z_bin_heights : np.ndarray
    delta_bin_centers : np.ndarray
    delta_bin_heights : np.ndarray
    n_turns_tot : int 
    nturns_profile_accumulation_interval : int
    index : int = 0 # index to keep track on where to append

    @classmethod
    def init_monitor(cls, 
                     num_z_bins : int, 
                     n_turns_tot: int, 
                     nturns_profile_accumulation_interval : int) -> Self:  # noqa: F821
        """Initialize dataclass with empty arrays containing zeta bin edges and zeta bin heights for every turn"""
        n_profiles = int(n_turns_tot / nturns_profile_accumulation_interval)
        return cls(
            z_bin_centers = np.zeros(num_z_bins, dtype=float),
            z_bin_heights = np.zeros([num_z_bins, n_profiles], dtype=float),
            delta_bin_centers = np.zeros(num_z_bins, dtype=float),
            delta_bin_heights = np.zeros([num_z_bins, n_profiles], dtype=float),
            n_turns_tot = n_turns_tot,
            nturns_profile_accumulation_interval = nturns_profile_accumulation_interval
        )

    def convert_zetas_and_stack_histogram(self, 
                                          zetas : Zeta_Container, 
                                          num_z_bins : int, 
                                          z_range : tuple,
                                          delta_range : tuple):
        # Convert data class of particles to histogram
        z = zetas.zeta.flatten()
        delta = zetas.delta.flatten()
        s = zetas.state.flatten()

        # Aggregate longitudinal coordinates of particles still alive
        zetas_accumulated = z[s>0]
        bin_heights, bin_borders = np.histogram(zetas_accumulated, bins=num_z_bins, range=(z_range[0], z_range[1]),)
        bin_widths = np.diff(bin_borders)
        bin_centers = bin_borders[:-1] + bin_widths / 2

        # Aggregate delta coordinates of particles still alive
        deltas_accumulated = delta[s>0]
        bin_heights_delta, bin_borders_delta = np.histogram(deltas_accumulated, bins=num_z_bins, range=(delta_range[0], delta_range[1]),)
        bin_widths_delta = np.diff(bin_borders_delta)
        bin_centers_delta = bin_borders_delta[:-1] + bin_widths_delta / 2

        # Append zeta bin centers and width only once
        if self.index == 0:
            self.z_bin_centers = bin_centers
            self.z_bin_widths = bin_widths
            self.delta_bin_centers = bin_centers_delta
            self.delta_bin_widths = bin_widths_delta


        # Append bin heights, if index is good
        if self.index < len(self.z_bin_heights[0]):
            self.z_bin_heights[:, self.index] = bin_heights
            self.delta_bin_heights[:, self.index] = bin_heights_delta
            #print(f'\nFilled in zeta profile number {self.index + 1} out of {len(self.z_bin_heights[0])}')
        else:
            print(f'\nIndex {self.index} above pre-specified number of profiles {len(self.z_bin_heights[0])}!\n')

        # Move to next step
        self.index += 1


    
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
    exn: np.ndarray
    eyn: np.ndarray
    sigma_delta: np.ndarray
    bunch_length: np.ndarray
    Nb: np.ndarray
    state: np.ndarray
    which_context : str
    full_data_turn_ind: np.ndarray
    includes_WS_profile_data : bool = False

    def update_at_turn(self, turn: int, parts: xp.Particles, twiss: xt.TwissTable):
        """Automatically update the records at given turn from the xpart.Particles."""

        # Store the particle ensemble quantities
        self.exn[turn] = _geom_epsx(parts, twiss) * parts.beta0[0] * parts.gamma0[0]
        self.eyn[turn] = _geom_epsy(parts, twiss) * parts.beta0[0] * parts.gamma0[0]
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
            exn=np.zeros(n_turns, dtype=float),
            eyn=np.zeros(n_turns, dtype=float),
            Nb=np.zeros(n_turns, dtype=float),
            sigma_delta=np.zeros(n_turns, dtype=float),
            bunch_length=np.zeros(n_turns, dtype=float),
            which_context=which_context,
            full_data_turn_ind = full_data_turn_ind
        )
    
    def to_dict(self):
        return {
            'exn': self.exn,
            'eyn': self.eyn,
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
            exn=np.hstack((self.exn[:1], self.exn[-1:])),
            eyn=np.hstack((self.eyn[:1], self.eyn[-1:])),
            sigma_delta=np.hstack((self.sigma_delta[:1], self.sigma_delta[-1:])),
            bunch_length=np.hstack((self.bunch_length[:1], self.bunch_length[-1:])),
            Nb=np.hstack((self.Nb[:1], self.Nb[-1:])),
            state=np.hstack((self.state[:, :1], self.state[:, -1:])),
            which_context=self.which_context
        )

    def append_WS_profile_monitor_data(self,
                                       monitorH_x_grid, 
                                       monitorH_x_intensity,
                                       monitorV_y_grid, 
                                       monitorV_y_intensity):
        """
        If tracking has been done with installed beam profile monitors, append data
        and save to json file
        """
        self.monitorH_x_grid = monitorH_x_grid.tolist()
        self.monitorH_x_intensity = monitorH_x_intensity.tolist()
        self.monitorV_y_grid = monitorV_y_grid.tolist()
        self.monitorV_y_intensity = monitorV_y_intensity.tolist()

        self.includes_WS_profile_data = True


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
            'exn': self.exn.tolist(),
            'eyn': self.eyn.tolist(),
            'sigma_delta': self.sigma_delta.tolist(),
            'bunch_length': self.bunch_length.tolist(),
            'Nb': self.Nb.tolist(),
            'state': self.state.tolist(),
            'which_context': self.which_context,
            'full_data_turn_ind': self.full_data_turn_ind.tolist()
        }
        if self.includes_WS_profile_data:
            print('\nAppending WS profile data to dictionary\n')
            data['monitorH_x_grid'] = self.monitorH_x_grid
            data['monitorH_x_intensity'] = self.monitorH_x_intensity
            data['monitorV_y_grid'] = self.monitorV_y_grid
            data['monitorV_y_intensity'] = self.monitorV_y_intensity
            del data['x'], data['y'], data['px'], data['py'] # delete as we have full transverse picture anyway

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
            exn=np.array(data['exn']),
            eyn=np.array(data['eyn']),
            sigma_delta=np.array(data['sigma_delta']),
            bunch_length=np.array(data['bunch_length']),
            Nb=np.array(data['Nb']),
            state=np.array(data['state']),
            which_context=data['which_context'],
            full_data_turn_ind=np.array(data['full_data_turn_ind'])
        )
    

    @classmethod
    def dict_from_json(cls, file_path):
        """
        Load the data from a JSON file and construct a dictionary from data
        """
        with open(file_path, 'r') as f:
            tbt_dict = json.load(f)

        return tbt_dict