"""
Main class containing the tune ripple simulator for SPS
- inspired by xtrack example: https://xsuite.readthedocs.io/en/latest/fast_lattice_changes.html
"""
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from pathlib import Path
import json
import os
import NAFFlib

import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo

from .beam_parameters import BeamParameters_SPS
from .sequences import SPS_sequence_maker
from .fma_ions import FMA

sequence_path = Path(__file__).resolve().parent.joinpath('../data/sps_sequences').absolute()

@dataclass
class Tune_Ripple_SPS:
    """
    Class to simulate tune ripple by varying quadrupole strengths
    
    Parameters:
    -----------
    qx0 : float
        horizontal tune
    qy0 : float
        vertical tune
    beta_beat: float
        relative beta beat, i.e. relative difference between max beta function and max original beta function
    use_symmetric_lattice: bool
        flag to use symmetric lattice without QFA and QDA
    n_linear : int
        default number of points if uniform linear grid for normalized X and Y are used
    r_min : float
        minimum radial distance from beam center to generate particles, to avoid zero amplitude oscillations for FMA
    n_sigma : float
        max number of beam sizes sigma to generate particles
    n_part : int
        number of macroparticles for beam generation
    z0 : float
        initial longitudinal offset zeta
    output_folder : str
        location to save data
    """
    qx0: float = 26.30
    qy0: float = 26.25
    beta_beat: float = None
    use_symmetric_lattice: bool = True
    ripple_period: int = 2000 
    num_turns: int = 10000
    n_linear: int = 100
    r_min: float = 0.1
    n_sigma: float = 10.0
    n_part : int = 5000
    z0: float = 0.0
    output_folder: str = 'output_tune_ripple'
    
    
    def _get_initial_normalized_coord_at_start(self, make_single_Jy_trace=True, y_norm0=0.05):
        """
        Return normalized coordinates of particle object, conforming class parameters
        Stores normalized x and y internally
        
        Parameters
        ----------
        make_single_Jy_trace : bool
            flag to create single trace with unique vertical action Jy, 
            with varying action Jx. "Trace" instead of "grid", if uniform beam is used
        y_norm0 : float
            starting normalized Y coordinate for the single Jy trace ):
    
        Returns:
        -------
        None
        """
        # Generate arrays of normalized coordinates 
        x_values = np.linspace(self.r_min, self.n_sigma, num=self.n_linear)  
        y_values = np.linspace(self.r_min, self.n_sigma, num=self.n_linear)  
    
        # Select single trace, or create a meshgrid for the uniform beam distribution
        if make_single_Jy_trace: 
            x_norm = x_values
            y_norm = y_norm0 * np.ones(len(x_norm))
            print('Making single-trace particles object with length {}\n'.format(len(y_norm)))
        else:
            X, Y = np.meshgrid(x_values, y_values)
            x_norm, y_norm = X.flatten(), Y.flatten()
        
        # Store initial normalized coordinates
        self._x_norm, self._y_norm = x_norm, y_norm
        
    
    def get_k_ripple_signal(self, k_amplitude=1e-6):
        """
        Generate simple noise signal on top of kqf/kqd values, for a given ripple period and amplitude
        
        Parameters:
        -----------
        k_amplitude : float
            ripple amplitude for kqf and kqd --> obtained from normalized FFT spectrum of IQD and IQF

        Returns:
        --------
        k_ripple_values : np.ndarray
            focusing quadrupole values corresponding to modulate Qx according to dq (if chosen plane)
        """

        turns = np.arange(1, self.num_turns+1)
        k_ripple = k_amplitude * np.sin(2 * np.pi * turns / self.ripple_period)
        
        print('Generated kqf/kqd ripple of amplitude {:.3e} with ripple period {}'.format(k_amplitude, self.ripple_period))
            
        return k_ripple
    

    def get_k_ripple_summed_signal(self, ripple_periods, kqf_amplitudes, kqd_amplitudes,
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

        turns = np.arange(1, self.num_turns+1)
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


    def find_k_from_q_setvalue(self, dq=0.05, plane='X'):
        """
        For desired tune amplitude modulation dQx or dQy, find corresponding change in quadrupole strengths
        'qh_setvalue' and 'qv_setvalue' are linear knobs that regulate QD and QF strength
        - https://gitlab.cern.ch/acc-models/acc-models-sps/-/blame/2021/toolkit/macro.madx?ref_type=heads#L81
        
        Parameters:
        -----------
        dq : float
            absolute change in tune amplitude, e.g. 0.05
        plane : str
            'X' or 'Y' (default is 'X')
        
        Returns:
        --------
        kqf_vals : np.ndarray
            focusing quadrupole values corresponding to modulate Qx according to dq (if chosen plane)
        kqd_vals : np.ndarray
            defocusing quadrupole values corresponding to modulate Qy according to dq (if chosen plane)
        turns : np.ndarray
            range over turns the amplitude modulation corresponds to 
        """
        # Load MADX line of SPS and define quad knobs
        sps_seq = SPS_sequence_maker(qx0=self.qx0, qy0=self.qy0)
        madx = sps_seq.load_madx_SPS()
        madx.exec('sps_define_quad_knobs')
        
        # Find old magnet stregths
        kqf_0 = madx.globals['kqf']
        kqd_0 = madx.globals['kqd']
        print('\nOld strengths: kqf = {:.5f}, kqd = {:.5f}'.format(kqf_0, kqd_0))
        
        # Adjust the qh_setvalue or qv_setvalue
        if plane == 'X':
            madx.input('qh_setvalue = {};'.format(sps_seq.qx0 + dq))
        elif plane == 'Y':
            madx.input('qv_setvalue = {};'.format(sps_seq.qy0 + dq))
        else:
            raise ValueError('Undefined plane!')
        twiss = madx.twiss().summary
        print('New tunes: Qx = {:.6f} and Qy = {:.6f}'.format(twiss['q1'], twiss['q2']))
        
        # Find new quadrupole strength
        kqf_1 = madx.globals['kqf']
        kqd_1 = madx.globals['kqd']
        print('New strengths: kqf = {:.5f}, kqd = {:.5f}\n'.format(kqf_1, kqd_1))
        
        # Reset the qh_setvalue or qv_setvalue
        if plane == 'X':
            madx.input('qh_setvalue = {};'.format(sps_seq.qx0))
        elif plane == 'Y':
            madx.input('qv_setvalue = {};'.format(sps_seq.qy0))
        print('Q setvalue reset!')
        
        # Find amplitudes
        amp_kqf = np.abs(kqf_1 - kqf_0)
        amp_kqd = np.abs(kqd_1 - kqd_0)
        
        # Create arrays of quadrupole strengths to iterate over
        turns = np.arange(1, self.num_turns+1)
        kqf_vals = kqf_0 + amp_kqf * np.sin(2 * np.pi * turns / self.ripple_period)
        kqd_vals = kqd_0 + amp_kqd * np.sin(2 * np.pi * turns / self.ripple_period)
        
        return kqf_vals, kqd_vals, turns
        
      
    def find_k_from_xtrack_matching(self, dq=0.05, nr_matches=10, use_symmetric_lattice=False, plane='both', show_plot=False,
                                    proton_optics='q26'):
        """
        Find desired tune amplitude modulation dQx or dQy by matching the global
        variable kqf and kqd
        
        Parameters:
        -----------
        dq : float
            absolute change in tune amplitude, e.g. 0.05
        nr_matches : int
            number of times to match Qx and Qy with propoer kqf and kqd, for interpolation resolution
        use_symmetric_lattice : bool
            flag to use symmetric lattice without QFA and QDA
        plane : str
            'X' or 'Y' or 'both' (default is 'X')
        
        Returns:
        --------
        kqf_vals : np.ndarray
            focusing quadrupole values corresponding to modulate Qx according to dq (if chosen plane)
        kqd_vals : np.ndarray
            defocusing quadrupole values corresponding to modulate Qy according to dq (if chosen plane)
        turns : np.ndarray
            range over turns the amplitude modulation corresponds to 
        """
        # Load Xsuite line with deferred expressions from MADx
        sps = SPS_sequence_maker(qx0=self.qx0, qy0=self.qy0, proton_optics=proton_optics)
        
        if use_symmetric_lattice:
            line, twiss = sps.load_SPS_line_with_deferred_madx_expressions(use_symmetric_lattice=use_symmetric_lattice)
        else:
            line, twiss = sps.load_xsuite_line_and_twiss(deferred_expressions=True)
        
        # Empty arrays of quadrupolar strenghts:
        kqfs = np.zeros(nr_matches)
        kqds = np.zeros(nr_matches)
        Qx_vals = np.zeros(nr_matches)
        Qy_vals = np.zeros(nr_matches)
    
        # Investigate linear dependence
        if plane == 'X':
            dqx, dqy = dq, 0.0
        elif plane == 'Y':
            dqx, dqy = 0.0, dq
        elif plane == 'both':
            dqx, dqy = dq, dq
        else:
            raise ValueError('Plane not valid - X or Y!')
            
        Qx_target = np.linspace(np.round(twiss['qx'], 2), np.round(twiss['qx'], 2) + dqx, num=nr_matches)
        Qy_target = np.linspace(np.round(twiss['qy'], 2), np.round(twiss['qy'], 2) + dqy, num=nr_matches)
        
        for i, Q in enumerate(Qx_target): 
        
            print('\nMatching Qx = {}, Qy = {}'.format(Qx_target[i], Qy_target[i]))
            
            # Match tunes to assigned values
            line.match(
                vary=[
                    xt.Vary('kqf', step=1e-8),
                    xt.Vary('kqd', step=1e-8),
                ],
                targets = [
                    xt.Target('qx', Qx_target[i], tol=1e-7),
                    xt.Target('qy', Qy_target[i], tol=1e-7)])
            
            twiss = line.twiss()
            
            # Add first values 
            kqfs[i] = line.vars['kqf']._value
            kqds[i] = line.vars['kqd']._value
            Qx_vals[i] = twiss['qx']
            Qy_vals[i] = twiss['qy']
            
            
        # Plot tune evolution over time
        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        ax[0].plot(Qx_vals, kqfs, '-', color='blue')
        ax[1].plot(Qy_vals, kqds, '-', color='red')
        ax[0].set_xlabel('$Q_{x}$')
        ax[0].set_ylabel('kqf')
        ax[1].set_xlabel('$Q_{y}$')
        ax[1].set_ylabel('kqd')
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if show_plot:
            plt.show()
        plt.close()
            
        # Find amplitudes
        amp_kqf = kqfs[-1] - kqfs[0]
        amp_kqd = kqds[-1] - kqds[0]
        
        # Create arrays of quadrupole strengths to iterate over
        turns = np.arange(1, self.num_turns+1)
        kqf_vals = kqfs[0] + amp_kqf * np.sin(2 * np.pi * turns / self.ripple_period)
        kqd_vals = kqds[0] + amp_kqd * np.sin(2 * np.pi * turns / self.ripple_period)
        
        # Make dictionary
        k_dict = {'Qx': Qx_vals.tolist(),
                  'Qy': Qy_vals.tolist(),
                  'kqf': kqfs.tolist(),
                  'kqd': kqds.tolist()}
        
        # Save dictionary to json file
        sym_string = '_symmetric_lattice' if use_symmetric_lattice else '_nominal_lattice'
        k_val_path = '{}/qx_{}_qy_{}/k_knobs'.format(sequence_path, self.qx0, self.qy0)
        os.makedirs(k_val_path, exist_ok=True)
        print('\nSaving k strengths to \n{}dq{}percent.json"\n'.format(k_val_path, int(100*dq)))
        print('Ripple period: {}'.format(self.ripple_period))
        
        with open("{}/k_vals_{}{}_dq{}percent.json".format(k_val_path, plane, sym_string, int(100*dq)), "w") as fp:
            json.dump(k_dict , fp) 
        
        return kqf_vals, kqd_vals, turns
        
        
    def load_k_from_xtrack_matching(self, dq=0.05, use_symmetric_lattice=False, plane='both'):
        """
        Parameters:
        -----------
        dq : float
            absolute change in tune amplitude, e.g. 0.05
        nr_matches : int
            number of times to match Qx and Qy with propoer kqf and kqd, for interpolation resolution
        use_symmetric_lattice : bool
            flag to use symmetric lattice without QFA and QDA
        plane : str
            'X' or 'Y' or 'both' (default is 'X')

        Returns:
        --------
        kqf_vals : np.ndarray
            focusing quadrupole values corresponding to modulate Qx according to dq (if chosen plane)
        kqd_vals : np.ndarray
            defocusing quadrupole values corresponding to modulate Qy according to dq (if chosen plane)
        turns : np.ndarray
            range over turns the amplitude modulation corresponds to 
        """
        # Trying loading knobs if exist already
        try:
            sym_string = '_symmetric_lattice' if use_symmetric_lattice else '_nominal_lattice'
            k_val_path = '{}/qx_{}_qy_{}/k_knobs'.format(sequence_path, self.qx0, self.qy0)
            
            with open("{}/k_vals_{}{}_dq{}percent.json".format(k_val_path, plane, sym_string, int(100*dq)), "r") as fp:
                k_dict = json.load(fp) 
            print('Loaded k_strength json file dq = {} percent\n'.format(int(100*dq)))
                
            # Find amplitudes
            amp_kqf = k_dict['kqf'][-1] - k_dict['kqf'][0]
            amp_kqd = k_dict['kqd'][-1] - k_dict['kqd'][0]
            
            # Create arrays of quadrupole strengths to iterate over
            turns = np.arange(1, self.num_turns+1)
            kqf_vals = k_dict['kqf'][0] + amp_kqf * np.sin(2 * np.pi * turns / self.ripple_period)
            kqd_vals = k_dict['kqd'][0] + amp_kqd * np.sin(2 * np.pi * turns / self.ripple_period)
            
        except FileNotFoundError:
            print('Did not find k_strength json file - creating new!\n')
            
            kqf_vals, kqd_vals, turns = self.find_k_from_xtrack_matching(dq, plane=plane, use_symmetric_lattice=use_symmetric_lattice)
    
        return kqf_vals, kqd_vals, turns
    
    
    def run_simple_ripple_with_twiss(self, dq=0.05, plane='X', use_xtrack_matching=True):
        """
        Run SPS standard tune ripple, with twiss command every turn to check tunes
            
        Parameters:
        -----------
        dq : float
            absolute change in tune amplitude, e.g. 0.05
        plane : str
            'X' or 'Y' or 'both (default is 'X')
        use_xtrack_matching : bool
            flag to use xtrack or MADX qh_setvalue for the matching
        
        Returns:
        --------
        turns : np.ndarray
            range over turns the amplitude modulation corresponds to 
        Qx : np.ndarray
            Qx tunes over corresponding tune range
        Qy : np.ndarray
            Qy tunes over corresponding tune range
        """
        
        # Get SPS Pb line with deferred expressions
        sps = SPS_sequence_maker(qx0=self.qx0, qy0=self.qy0)
        line, twiss = sps.load_xsuite_line_and_twiss(deferred_expressions=True)
        if use_xtrack_matching:
            kqf_vals, kqd_vals, turns = self.load_k_from_xtrack_matching(dq=dq, plane=plane)
        else:   
            kqf_vals, kqd_vals, turns = self.find_k_from_q_setvalue(dq=dq, plane=plane)
        
        # Generate particles
        fma_sps = FMA(n_linear=self.n_linear, r_min=self.r_min, n_sigma=self.n_sigma, z0=self.z0)
        particles = fma_sps.generate_particles(line, BeamParameters_SPS, make_single_Jy_trace=True)
        
        
        # Empty array for turns
        Qx = np.zeros(len(kqf_vals))
        Qy = np.zeros(len(kqf_vals))
        
        for ii in range(self.num_turns):
            if ii % 20 == 0: print(f'Turn {ii} of {self.num_turns}')
        
            # Change the strength of the quads
            line.vars['kqf'] = kqf_vals[ii]
            line.vars['kqd'] = kqd_vals[ii]
        
            # Track one turn
            line.track(particles)
        
            # Check tunes
            twiss = line.twiss()
            Qx[ii] = twiss['qx']
            Qy[ii] = twiss['qy']

        # Plot tune evolution over time
        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        fig.suptitle('Xtrack matching') if use_xtrack_matching else fig.suptitle('MADX qh_setvalue')
        ax[0].plot(turns, Qx, '-', color='blue', label='$Q_{x}$')
        ax[1].plot(turns, Qy, '-', color='red', label='$Q_{y}$')
        ax[0].set_ylabel('Tune')
        ax[0].set_xlabel('Turns')
        ax[1].set_xlabel('Turns')
        
        plt.show()
        
        return turns, Qx, Qy
    
    
    def run_ripple(self, dq=0.05, plane='X',
                   load_tbt_data=False,
                   save_tbt_data=True,
                   make_single_Jy_trace=True,
                   use_Gaussian_beam=False,
                   use_symmetric_lattice=False,
                   install_SC_on_line = True,
                   sextupolar_value_to_add=None,
                   beta_beat=None,
                   add_non_linear_magnet_errors=False,
                   plane_beta_beat='Y',
                   vary_tune=True
                   ):
        """
        Test SPS tune ripple during tracking
            
        Parameters:
        -----------
        dq : float
            absolute change in tune amplitude, e.g. 0.05
        plane : str
            'X' or 'Y' (default is 'X') in which to perform ripple
        load_tbt_data : bool
            flag to load data if already tracked
        save_tbt_data : bool
            flag to save tracking data
        make_single_Jy_trace : bool
            flag to create single trace with unique vertical action
        use_Gaussian_beam : bool
            if not uniform beam or single Jy trace, use Gaussian beam for tracking
        use_symmetric_lattice : bool
            flag to use symmetric lattice without QFA and QDA
        install_SC_on_line : bool
            flag to install space charge on line with FMA ions
        sextupolar_value_to_add : float, optional
            k2 value of one extraction sextupole in SPS, if not None
        beta_beat : float 
            relative difference in beta functions (Y for SPS)
        add_non_linear_magnet_errors : bool
            whether to add non-linear chromatic errors for SPS
        plane_beta_beat : str
            'X' or 'Y' (default is 'X') - in which plane to perform desired beta-beat
        vary_tune : bool
            flag whether to run with ripple (i.e. vary tune) or without as control case
        
        Returns:
        --------
        x, y, px, py : np.ndarrays
            numpy arrays with turn-by-turn data
        """
        # Get SPS Pb line with deferred expressions
        sps = SPS_sequence_maker(qx0=self.qx0, qy0=self.qy0)
        line, twiss = sps.load_SPS_line_with_deferred_madx_expressions(use_symmetric_lattice=use_symmetric_lattice, 
                                                                       add_non_linear_magnet_errors=add_non_linear_magnet_errors)
        
        # If sextupolar value is set, set this value
        if sextupolar_value_to_add is not None:
            line = self._set_LSE_extraction_sextupolar_value(line, k2_value=sextupolar_value_to_add)
            print('\nSetting klse10602 sextupolar value to {}\n'.format(line.vars['klse10602']._value))
        
        # Add beta-beat if desired 
        if beta_beat is not None:
            sps_seq = SPS_sequence_maker()
            line = sps_seq.generate_xsuite_seq_with_beta_beat(beta_beat=beta_beat, line=line, plane=plane_beta_beat)

        if install_SC_on_line:
            fma_sps = FMA()
            line = fma_sps.install_SC_and_get_line(line, BeamParameters_SPS(), optimize_for_tracking=False)
            print('Installed space charge on line\n')
        kqf_vals, kqd_vals, turns = self.load_k_from_xtrack_matching(dq=dq, plane=plane)

        # Generate particles
        fma_sps = FMA(n_linear=self.n_linear, r_min=self.r_min, n_sigma=self.n_sigma)
        
        # Use Gaussian beam if desired, or uniform beam in action space
        if use_Gaussian_beam: 
            particles = SPS_sequence_maker.generate_SPS_gaussian_beam(line, n_part=self.n_part)
            print('\nGenerated Gaussian beam!\n')
        else:
            particles = fma_sps.generate_particles(line, BeamParameters_SPS, make_single_Jy_trace=make_single_Jy_trace)
            print('\nGenerated uniform beam, with single Jy trace: {}\n'.format(make_single_Jy_trace))
        
        # Empty array for turns
        x = np.zeros([len(particles.x), self.num_turns]) 
        y = np.zeros([len(particles.y), self.num_turns])
        px = np.zeros([len(particles.px), self.num_turns]) 
        py = np.zeros([len(particles.py), self.num_turns])
        state = np.zeros([len(particles.state), self.num_turns])
        
        # Track the particles and return turn-by-turn coordinates
        for ii in range(self.num_turns):
            if ii % 20 == 0: print(f'Turn {ii} of {self.num_turns}')
            
            x[:, ii] = particles.x
            y[:, ii] = particles.y
            px[:, ii] = particles.px
            py[:, ii] = particles.py
            state[:, ii] = particles.state
        
            # Change the strength of the quads
            if vary_tune:
                line.vars['kqf'] = kqf_vals[ii]
                line.vars['kqd'] = kqd_vals[ii]
        
            # Track one turn
            line.track(particles)
        
        # Set particle trajectories of dead particles that got lost in tracking
        self._state = state
        
        if save_tbt_data:
            os.makedirs(self.output_folder, exist_ok=True)
            np.save('{}/x.npy'.format(self.output_folder), x)
            np.save('{}/y.npy'.format(self.output_folder), y)
            np.save('{}/px.npy'.format(self.output_folder), px)
            np.save('{}/py.npy'.format(self.output_folder), py)
            np.save('{}/state.npy'.format(self.output_folder), state)
            print('Saved tracking data.')
    
        return x, y, px, py
    
    
    def load_tracking_data(self):
        """
        Loads numpy data if tracking has already been made
        
        Returns:
        --------
        x, y, px, py : np.ndarrays
            numpy arrays with turn-by-turn data
        """
        try:            
            x=np.load('{}/x.npy'.format(self.output_folder))
            y=np.load('{}/y.npy'.format(self.output_folder))
            px=np.load('{}/px.npy'.format(self.output_folder))
            py=np.load('{}/py.npy'.format(self.output_folder))
            self._state=np.load('{}/state.npy'.format(self.output_folder))
            return x, y, px, py

        except FileNotFoundError:
            raise FileNotFoundError('Tracking data does not exist - set correct path or generate the data!')
    
    
    def load_tune_data(self):
        """
        Loads numpy data of tunes if FMA has already been done
        
        Returns:
        --------
        Qx, Qy : np.ndarrays
            numpy arrays with turn-by-turn tune data
        """
        try:
            Qx = np.load('{}/Qx.npy'.format(self.output_folder))
            Qy = np.load('{}/Qy.npy'.format(self.output_folder))
            return Qx, Qy
        except FileNotFoundError:
            raise FileNotFoundError('Tune data does not exist - set correct path or perform FMA!')

    
    def get_tune_from_tbt_data(self, x_tbt_data, y_tbt_data, 
                               k, Qmin=0.0, save_tune_data=True):
        """
        Use NAFF
        
        Parameters:
        ----------
        x_tbt_data, y_tbt_data : numpy.ndarrays 
            arrays of turn-by-turn data for particles 
        Qmin : float, optional 
            if desired, filter out some lower frequencies
        k : int
            length of period to evaluate tune over, "tune period"
        save_tune_data : bool
            flag to save tune data
        
        Returns:
        --------
        Qx, Qy : np.ndarrays
            numpy arrays with turn-by-turn tune data
        --------
        """
        # Calculate the index "window" from which tunes are extracted 
        L = len(x_tbt_data[0]) # number of turns
        
        Qx = np.zeros([len(x_tbt_data), L - k + 1])
        Qy = np.zeros([len(x_tbt_data), L - k + 1])
        
        # Iterate over particles to find tune
        for i_part in range(len(x_tbt_data)):
            
            if i_part % 2 == 0:
                print('NAFF algorithm of particle {}'.format(i_part))
                        
            # Iterate over subwindows of length k to find tune
            for i in range(L - k + 1):
                
                if i_part % 2 == 0 and i % 4000 == 0:
                    print('Tune after turn {}'.format(i))
                    
                # Find dominant frequency with NAFFlib - also remember to subtract mean 
                Qx_raw = NAFFlib.get_tunes(x_tbt_data[i_part, i:i+k] \
                                                - np.mean(x_tbt_data[i_part, i:i+k]), 2)[0]
                Qx[i_part, i] = Qx_raw[np.argmax(Qx_raw > Qmin)]  # find most dominant tune larger than this value
                Qy_raw = NAFFlib.get_tunes(y_tbt_data[i_part, i:i+k] \
                                                - np.mean(y_tbt_data[i_part, i:i+k]), 2)[0]
                Qy[i_part, i] = Qy_raw[np.argmax(Qy_raw > Qmin)]
                
        
        # Change all zero-valued tunes to NaN
        Qx[Qx == 0.0] = np.nan
        Qy[Qy == 0.0] = np.nan

        if save_tune_data:
            os.makedirs(self.output_folder, exist_ok=True)
            np.save('{}/Qx.npy'.format(self.output_folder), Qx)
            np.save('{}/Qy.npy'.format(self.output_folder), Qy)

        return Qx, Qy
    
    
    def run_ripple_and_analysis(self, dq=0.05, plane='X',
                   load_tbt_data=False,
                   make_single_Jy_trace=True,
                   action_limits=None,
                   use_symmetric_lattice=False,
                   install_SC_on_line=True,
                   beta_beat=None,
                   add_non_linear_magnet_errors=False,
                   num_particles_to_plot=10,
                   sextupolar_value_to_add=None,
                   plot_random_colors=False,
                   action_in_logscale=False,
                   also_show_plot=False,
                   plot_dead_particles=False,
                   phase_sweep_up_to_turn=None,
                   phase_space_sweep_interval=1000,
                   plane_beta_beat='Y'
                   ):
        """
        Run SPS tune ripple wtih tracking and generate phase space plots
            
        Parameters:
        -----------
        dq : float
            amplitude of oscillations in chosen plane
        plane : str
            'X' or 'Y'
        load_tbt_data : bool
            flag to load data if already tracked
        make_single_Jy_trace : bool 
            flag to create single trace with unique vertical action
        action_limits: list, optional
            list with [J_min, J_max] to include particles in fixed action range. Default: None, 
            meaning plotting "num_particles_to_plot" evenly spread out
        use_symmetric_lattice : bool
            flag to use symmetric lattice without QFA and QDA
        install_SC_on_line : bool
            flag to install space charge on line with FMA ions
        beta_beat : float 
            relative difference in beta functions (Y for SPS)
        add_non_linear_magnet_errors : bool
            whether to add non-linear chromatic errors for SPS
        num_particles_to_plot : int
            number of particles to include in plot
        sextupolar_value_to_add : float, optional
            k2 value of one extraction sextupole in SPS, if not None
        plot_random_colors : bool, optional
            plots random colors for each particle if True, otherwise colormap depending on starting amplitude
        action_in_logscale: bool, optional
            whether to plot action in logscale or not
        plot_dead_particles : bool, optional
            whether to plot particles that got lost in tracking or not
        phase_space_sweep_interval : int
            number of steps to follow the evolution of phase space
        plane_beta_beat : str
            'X' or 'Y' (default is 'X') - in which plane to perform desired beta-beat
        
        Returns:
        --------
        """
        if load_tbt_data:
            x, y, px, py = self.load_tracking_data()
        else:
            x, y, px, py = self.run_ripple(dq=dq, plane=plane, make_single_Jy_trace=make_single_Jy_trace, use_symmetric_lattice=use_symmetric_lattice,
                                           install_SC_on_line=install_SC_on_line, sextupolar_value_to_add=sextupolar_value_to_add,
                                           beta_beat=beta_beat, add_non_linear_magnet_errors=add_non_linear_magnet_errors,
                                           plane_beta_beat=plane_beta_beat)
                
        # Load relevant SPS line and twiss
        self._get_initial_normalized_coord_at_start() # loads normalized coord of starting distribution
        sps = SPS_sequence_maker(qx0=self.qx0, qy0=self.qy0)
        line, twiss = sps.load_SPS_line_with_deferred_madx_expressions(use_symmetric_lattice=use_symmetric_lattice)
        
        # If sextupolar value is set, set this value
        if sextupolar_value_to_add is not None:
            line = self._set_LSE_extraction_sextupolar_value(line, k2_value=sextupolar_value_to_add)
            print('\nSetting klse10602 sextupolar value to {}\n'.format(line.vars['klse10602']._value))
        
        # Try to load tune-data
        k = int(np.ceil(2 / twiss['qs'])) # tune evaluated over two synchrotron periods
        try:
            Qx, Qy = self.load_tune_data()
        except FileNotFoundError:
            Qx, Qy = self.get_tune_from_tbt_data(x, y, k)
        
        # Load turns and quadrupole strengths
        kqf_vals, kqd_vals, turns = self.load_k_from_xtrack_matching(dq=dq, plane=plane)
        turns = np.array(turns)
        
        # Plot for phase space sweep
        if phase_sweep_up_to_turn is None:
            phase_sweep_up_to_turn = turns[-1]
        
        # Calculate normalized coordinates
        X = x / np.sqrt(twiss['betx'][0]) 
        PX = twiss['alfx'][0] / np.sqrt(twiss['betx'][0]) * x + np.sqrt(twiss['betx'][0]) * px
        Y = y / np.sqrt(twiss['bety'][0]) 
        PY = twiss['alfy'][0] / np.sqrt(twiss['bety'][0]) * y + np.sqrt(twiss['bety'][0]) * py
        
        # Calculate action for each particle
        Jx = X**2 + PX **2
        Jy = Y**2 + PY **2


        # Select particle index to plot - evenly spaced out in action
        if action_limits is None:
            ind = np.arange(start=1, stop=len(x), step=len(x) / num_particles_to_plot, dtype=int)
        else:
            ind = np.where((action_limits[0] < Jx[:, 0]) & (action_limits[1] > Jx[:, 0]))[0]
            print('\nCustom action index between {} < Jx < {}\n'.format(Jx[ind[0]], Jx[ind[-1]]))

        # Calculate phase space angle
        if plane == 'X':
            phi = np.arctan2(X, PX)          
        elif plane == 'Y':
            phi = np.arctan2(Y, PY)
        else:
            raise ValueError('Plane invalid - has to be "X" or "Y"')
        
        # Generate the plots
        J = Jx if plane == 'X' else Jy
        Q = Qx if plane == 'X' else Qy
        Z = X if plane == 'X' else Y
        PZ = PX if plane == 'X' else PY
        
        # Set values to nan if particles are dead 
        if not plot_dead_particles:
            J[self._state < 1.0] = np.nan
            Q[self._state[:, :len(x[0]) - k + 1] < 1.0] = np.nan
            Z[self._state < 1.0] = np.nan
            PZ[self._state < 1.0] = np.nan
            
            # Print number of lost particles in tracking
            last_state = self._state[:, -1]
            print('\nNumber of particles lost: {}\nError code: {}'.format(len(last_state[last_state < 1]), 
                                                                          self._state[self._state < 1]))

        # Plot sweeping phase space, and actions and tunes
        self.plot_sweeping_phase_space(turns, J, Q, ind, k, Z, PZ, action_in_logscale=action_in_logscale, 
                                       plot_up_to_turn=phase_sweep_up_to_turn, sweep_step=phase_space_sweep_interval)
        self.plot_action_and_tunes(turns, phi, J, Q, Z, PZ, ind, k, plot_random_colors=plot_random_colors,
                                      action_in_logscale=action_in_logscale,
                                      also_show_plot=also_show_plot, num_particles_to_plot=num_particles_to_plot)
    
    
    def run_ripple_with_Gaussian_beam(self, dq=0.05, plane='X',
                   load_tbt_data=False,
                   use_symmetric_lattice=False,
                   install_SC_on_line=True,
                   beta_beat=None,
                   add_non_linear_magnet_errors=False,
                   sextupolar_value_to_add=None,
                   plane_beta_beat='Y',
                   vary_tune=True
                   ):
                                      
                                      
        """ 
        FMA analysis of Gaussian distribution and tails with tune ripple

        dq : float
            amplitude of oscillations in chosen plane
        plane : str
            'X' or 'Y'
        load_tbt_data : bool
            flag to load data if already tracked
        use_symmetric_lattice : bool
            flag to use symmetric lattice without QFA and QDA
        install_SC_on_line : bool
            flag to install space charge on line with FMA ions
        beta_beat : float 
            relative difference in beta functions (Y for SPS)
        add_non_linear_magnet_errors : bool
            whether to add non-linear chromatic errors for SPS
        sextupolar_value_to_add : float, optional
            k2 value of one extraction sextupole in SPS, if not None
        vary_tune : bool
            flag whether to enact ripple (i.e. vary tune) or without as control case
            
        Returns:
        -------
        """
        if not vary_tune:
            print('Running tracking WITHOUT tune ripple!')    

        # Load turn-by-turn data or run the tracking with an external tune ripple
        if load_tbt_data:
            x, y, px, py = self.load_tracking_data()
        else:
            x, y, px, py = self.run_ripple(dq=dq, plane=plane, use_Gaussian_beam=True, use_symmetric_lattice=use_symmetric_lattice,
                                           install_SC_on_line=install_SC_on_line, sextupolar_value_to_add=sextupolar_value_to_add,
                                           beta_beat=beta_beat, add_non_linear_magnet_errors=add_non_linear_magnet_errors,
                                           plane_beta_beat=plane_beta_beat, vary_tune=vary_tune)
        
        # Load relevant SPS line and twiss
        self._get_initial_normalized_coord_at_start() # loads normalized coord of starting distribution
        sps = SPS_sequence_maker(qx0=self.qx0, qy0=self.qy0)
        line, twiss = sps.load_SPS_line_with_deferred_madx_expressions(use_symmetric_lattice=use_symmetric_lattice)
        
        # If sextupolar value is set, set this value
        if sextupolar_value_to_add is not None:
            line = self._set_LSE_extraction_sextupolar_value(line, k2_value=sextupolar_value_to_add)
            print('\nSetting klse10602 sextupolar value to {}\n'.format(line.vars['klse10602']._value))

        # FMA of tunes - load tune data if exists
        k = int(np.ceil(2 / twiss['qs'])) # tune evaluated over two synchrotron periods
        try:
            Qx, Qy = self.load_tune_data()
        except FileNotFoundError:
            Qx, Qy = self.get_tune_from_tbt_data(x, y, k)
        
        # Load turns and quadrupole strengths
        kqf_vals, kqd_vals, turns = self.load_k_from_xtrack_matching(dq=dq, plane=plane)
        turns = np.array(turns)
        
        # Calculate normalized coordinates
        X = x / np.sqrt(twiss['betx'][0]) 
        PX = twiss['alfx'][0] / np.sqrt(twiss['betx'][0]) * x + np.sqrt(twiss['betx'][0]) * px
        Y = y / np.sqrt(twiss['bety'][0]) 
        PY = twiss['alfy'][0] / np.sqrt(twiss['bety'][0]) * y + np.sqrt(twiss['bety'][0]) * py
        
        # Analysis of tails, and plot tune over beam position
        z_bar_tbt_data = X if plane=='X' else Y
        # HERE - convert instead to position normalized to BEAM SIZE - UPDATE THIS!
        self.plot_particle_distribution(z_bar_tbt_data, plane=plane)        
    

    def plot_particle_distribution(self, z_bar_tbt_data, plane='X', n_bins=100,
                                   also_show_plot=False):

        """
        Generates histogram of particle distribution in terms of beam sigma - at start and at the end

        Parameters:
        -----------
        z_bar_tbt_data : numpy.ndarray
            turn-by-turn data from tracking, either x or y
        plane : str
            'X' or 'Y'
        n_bins : int
            number of bins for histogram
        also_show_plot : bool
            whether to run 'plt.show()' at the end

        Returns:
        --------
        None
        """
        print('\nGenerating the particle distribution plot\n')
        # Generate figure object
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(z_bar_tbt_data[:, 0], bins=n_bins, histtype='step', color='b', alpha=0.6, density=True, label='Start') # first turn
        ax.hist(z_bar_tbt_data[:, -1], bins=n_bins, histtype='step', color='r', alpha=0.6, density=True, label='End') # last turn

                # Add correct labels        
        if plane == 'X':                       
            ax.set_xlabel('$x$ [mm]')
        elif plane == 'Y':
            ax.set_xlabel('$y$ [mm]')
        ax.set_ylabel('Normalized counts')
        ax.legend()
        
         # Save figure
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig.savefig('{}/Start_and_end_distribution.png'.format(self.output_folder), dpi=250)

        if also_show_plot:
            plt.show()


    def plot_action_and_tunes(self, 
                              turns,
                              phi,
                              J, 
                              Q,
                              Z,
                              PZ,
                              ind,
                              k,
                              plane='X',
                              plot_random_colors=False,
                              action_in_logscale=False,
                              also_show_plot=False,
                              num_particles_to_plot=10,
                              generate_stroboscopic_view=False):
        """
        Creates plots of action and tune evolution, and phase space
        
        Parameters:
        -----------
        turns : numpy.ndarray
            array containing turns from tracking
        phi : numpy.ndarray
            array with particle phase from tracking 
        J : numpy.ndarray
            array with particle actions from tracking 
        Q : numpy.ndarray
            array with particle tunes from tracking
        Z : numpy.ndarray
            array with normalized phase space coordinates X or Y
        PZ : numpy.ndarray
            array with normalized phase space coordinates PX or PY
        ind : numpy.ndarray
            indices of particles to plot
        k : int
            number of turns over which to evaluate the tune (normally two synchrotron periods)
        plane : str
            'X' or 'Y'
        plot_random_colors : bool, optional
            plots random colors for each particle if True, otherwise colormap depending on starting amplitude
        action_in_logscale: bool, optional
            whether to plot action in logscale or not
        num_particles_to_plot : int
            number of particles to include in plot
        generate_stroboscopic_view : bool
            flag whether to generate stroboscopic plots or not
        
        Returns:
        --------
        None
        """
        
        # Take colors from colormap of normalized phase space
        if plot_random_colors:
            colors = plt.cm.prism(np.linspace(0, 1, len(self._x_norm))) # varies according to prism, strongly 
        else:
            colors = plt.cm.cool(np.linspace(0, 1, len(self._x_norm)))

        # First make stroboscopic view if desired
        if generate_stroboscopic_view:
            self.generate_stroboscopic_view(turns, phi, J, ind, num_plots = 10, plane='X')

        ######### Action evolution and over time #########
        fig, ax = plt.subplots(2, 1, figsize=(11,7), sharex=True)
        fig.suptitle('Action and tune evolution - tune modulation period of {} turns'.format(self.ripple_period), fontsize=13)
        
        ######### NORMALIZED PHASE SPACE #########
        fig2, ax2 = plt.subplots(1, 1, figsize=(8,6))
        fig2.suptitle('Normalized phase space - tune modulation period of {} turns'.format(self.ripple_period), fontsize=14)
        
        ######### POLAR ACTION SPACE #########
        fig3, ax3 = plt.subplots(1, 1, figsize=(8,6))
        fig3.suptitle('Polar action space - tune modulation period of {} turns'.format(self.ripple_period), fontsize=14)
        
        print('\nStarting plotting\n')
        j = 0
        for particle in ind:
            
            if j % 10 == 0:
                print('Index {} out of {}'.format(j, len(ind))) 
            
            # Mix black and colorbar 
            color=colors[particle] if j % 2 == 0 else 'k'
                
            # Plot normalized phase space and action space           
            ax[0].plot(turns, J[particle, :], 'o', color=color, alpha=0.5, markersize=1.2)
            ax[1].plot(turns[k-1:], Q[particle, :], '-', color=color)
            ax2.plot(Z[particle, :], PZ[particle, :], 'o', color=color, alpha=0.5, markersize=1.5)
            ax3.plot(phi[particle, :], J[particle, :], 'o', color=color, alpha=0.5, markersize=1.2)

            ax[1].set_xlabel('Turns')
            ax3.set_xlabel(r"$\phi$ [rad]")

            j += 1

        # Add correct labels        
        if plane == 'X':                       
            ax[0].set_ylabel('$J_{x}$')
            ax[1].set_ylabel('$Q_{x}$')
            ax2.set_ylabel(r"$P_{x}$")
            ax2.set_xlabel(r"$X$")
            ax3.set_ylabel(r"$J_{x}$")     
        elif plane == 'Y':
            ax[0].set_ylabel('$J_{y}$')
            ax[1].set_ylabel('$Q_{y}$')
            ax2.set_ylabel(r"$P_{Y}$")
            ax2.set_xlabel(r"$Y$")
            ax3.set_ylabel(r"$J_{y}$")
    
        # Set logscale for action
        if action_in_logscale:
            ax[0].set_yscale('log')
            ax3.set_yscale('log')
        
        # Add colorbar, normalized to beam size (in sigmas)
        if not plot_random_colors:
            fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(min(self._x_norm), max(self._x_norm)), cmap='cool'),
                 ax=ax[0], orientation='vertical', label='$\sigma_{x}$')
            fig3.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(min(self._x_norm), max(self._x_norm)), cmap='cool'),
                 ax=ax3, orientation='vertical', label='$\sigma_{x}$')
        
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        
        # Save figures
        fig.savefig('{}/Action_and_tune_over_turns.png'.format(self.output_folder), dpi=250)
        fig2.savefig('{}/Normalized_phase_space.png'.format(self.output_folder), dpi=250)
        fig3.savefig('{}/Polar_action_space_{}.png'.format(self.output_folder, plane), dpi=250)
        
        if also_show_plot:
            plt.show()
    
    
    def plot_sweeping_phase_space(self, turns, J, Q, ind, k, Z, PZ, sweep_step=1000, plane='X', 
                                  action_in_logscale=False, plot_up_to_turn=None):
        """
        Create series of snap-shots in phase space over actions

        Parameters:
        -----------
        turns : numpy.ndarray
            array containing turns from tracking
        J : numpy.ndarray
            array with particle actions from tracking 
        Q : numpy.ndarray
            array with particle tunes from tracking
        ind : numpy.ndarray
            indices of particles to plot
        k : int
            number of turns over which to evaluate the tune (normally two synchrotron periods)
        Z : numpy.ndarray
            array with normalized phase space coordinates X or Y
        PZ : numpy.ndarray
            array with normalized phase space coordinates PX or PY
        plane : str
            'X' or 'Y'
        sweep_step : int
            interval between every frozen plot in phase space
        action_in_logscale: bool, optional
            whether to plot action in logscale or not
        plot_up_to_turn : int, optional
            until which turn to plot

        Returns:
        --------
        None
        """
        if plot_up_to_turn is None:
            plot_up_to_turn = turns[-1]
        
        # Take colors from colormap of normalized phase space
        colors = plt.cm.cool(np.linspace(0, 1, len(self._x_norm)))
    
        # Create folder for sweeping plot if does not exist
        output_sweep = '{}/phase_space_sweep'.format(self.output_folder)
        os.makedirs(output_sweep, exist_ok=True)
           
        sweep_ind = np.arange(start=1, stop=plot_up_to_turn, step=sweep_step)
             
        # Iterate over index sweep for stroboscopic view
        for i in range(len(sweep_ind)):
            
            if i % 5 == 0:
                print('Sweep nr {}'.format(i+1))
            
            ######### Stroboscopic action view #########
            #fig, ax = plt.subplots(2, 2, figsize=(13,7))
            fig = plt.figure(figsize=(13, 7), tight_layout=True)
            fig.suptitle('Phase space sweep - tune modulation every {} turns, {} turns in total'.format(self.ripple_period, self.num_turns), 
                         fontsize=13)    

            # Create axes for each purpose            
            gs = gridspec.GridSpec(2, 2)
            
            # Action and tune evolution over time
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax3 = fig.add_subplot(gs[:, 1]) # ax for normalized phase space 
        
            # Find maximum limit in phase space
            Zmax = np.max(Z)
            PZmax = np.max(PZ)
            max_lim = np.max([Zmax, PZmax])
            min_lim = -max_lim
                
            # Iterate over particles to plot
            j = 0
            for particle in ind:
                                
                # Mix black and colorbar 
                color=colors[particle] if j % 2 == 0 else 'k'
                    
                # Plot normalized phase space and action space
                ax1.plot(turns[:plot_up_to_turn], J[particle, :plot_up_to_turn], 'o', color=color, alpha=0.5, markersize=1.2)
                ax2.plot(turns[k-1:plot_up_to_turn], Q[particle, :plot_up_to_turn-k+1], 'o', color=color, alpha=0.5, markersize=1.2)
                if j == 0:
                    ax1.axvline(x=sweep_ind[i], lw=0.8, alpha=0.7, color='k')
                    ax2.axvline(x=sweep_ind[i], lw=0.8, alpha=0.7, color='k') 

                ax3.plot(Z[particle, :sweep_ind[i]], PZ[particle, :sweep_ind[i]], 'o', color=color, alpha=0.5, markersize=2.5)
                j += 1
        
            ax2.set_xlabel('Turns')
            plt.setp(ax1.get_xticklabels(), visible=False)
            
            ax3.set_xlim(min_lim, max_lim)
            ax3.set_ylim(min_lim, max_lim)
            
            # Add correct labels        
            if plane == 'X':                       
                ax1.set_ylabel('$J_{x}$')
                ax2.set_ylabel('$Q_{x}$')
                ax3.set_ylabel(r"$P_{x}$")
                ax3.set_xlabel(r"$X$")
            elif plane == 'Y':
                ax1.set_ylabel('$J_{y}$')
                ax2.set_ylabel('$Q_{y}$')
                ax3.set_ylabel(r"$P_{y}$")
                ax3.set_xlabel(r"$Y$")
        
            # Set logscale for action
            if action_in_logscale:
                ax1.set_yscale('log')
            
            fig.savefig('{}/{}_Phase_space_sweep.png'.format(output_sweep, i), dpi=250)
            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            
            del fig, ax1, ax2, ax3
            
            plt.close()

        
        
    
    def generate_stroboscopic_view(self, turns, phi, J, ind, num_plots = 10, plane='X'):
        """ 
        Create plots of action evolution, along with polar action space with one turn per modulation period
        Saves these plots in a subfolder
        
        Parameters:
        -----------
        turns: np.ndarray
            array with turn evolution
        phi: np.ndarray
            array with phase in phase space
        J: np.ndarray
            array of actions
        ind: np.ndarray with integers
            array containing index of particles to plot
        num_plots: int, optional
            number of stroboscopic plots.
        plane: str, optional
            'X' or 'Y'
            
        Returns:
        --------
        None
        """
        # Take colors from colormap of normalized phase space
        colors = plt.cm.cool(np.linspace(0, 1, len(self._x_norm)))
    
        # Create folder for stroboscopic plot if does not exist
        output_strobo = '{}/strobo'.format(self.output_folder)
        os.makedirs(output_strobo, exist_ok=True)
           
        # Step every time in number of terms
        strobo_step = int(self.ripple_period / num_plots)
        
        counter = 1
        # Iterate over index sweep for stroboscopic view
        for i in range(0, self.ripple_period, strobo_step):
        
            ######### Stroboscopic action view #########
            fig, ax = plt.subplots(1, 2, figsize=(11,7), sharey=True)
            fig.suptitle('Stroboscopic - tune modulation every {} turns, {} turns in total'.format(self.ripple_period, self.num_turns), 
                         fontsize=14)    
        
            # Stroboscopic turn index to plot
            ind_strob = np.arange(start=i, stop=self.num_turns+i, step=self.ripple_period, dtype=int)
            
            if counter % 2 == 0:
                print('\nMaking stroboplot nr {} out of {}'.format(counter, num_plots))
                print('Turn index: {}\n'.format(ind_strob))
            
            
            # Iterate over particles to plot
            j = 0
            for particle in ind:
                                
                # Mix black and colorbar 
                color=colors[particle] if j % 2 == 0 else 'k'
                    
                # Plot normalized phase space and action space
                ax[0].plot(phi[particle, ind_strob], J[particle, ind_strob], 'o', color=color, alpha=0.5, markersize=2.1) 
                ax[1].plot(turns, J[particle, :], 'o', color=color, alpha=0.5, markersize=1.2)
                for strob_turn in ind_strob:
                    ax[1].axvline(x=strob_turn, lw=0.8, alpha=0.7, color='k')                  
                j += 1
    
            # Add correct labels           
            ax[0].set_xlabel(r"$\phi$ [rad]")
            ax[1].set_xlabel('Turns')
            ax[0].set_ylabel('$J_{x}$') if plane == 'X' else ax[1].set_ylabel('$J_{y}$')
            ax[1].set_xlim(0, self.num_turns / 10)
            
            fig.savefig('{}/{}_Stroboscopic_Action.png'.format(output_strobo, counter), dpi=250)
            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            
            counter += 1
            del fig, ax
            
            plt.close()


    def _set_LSE_extraction_sextupolar_value(self, line, k2_value=0.1):
        """
        Add sextupolar component to one extraction LSE sextupole in SPS (normally zero-valued)
        
        Parameters:
        -----------
        line : xtrack.line
            xtrack line object to search through
        k2_value: float
            sextupolar component of first extraction sextupole
        
        Returns:
        --------
        line : xtrack.line
            xtrack line object with new sextupole values
        """
        # Adjust knob for sextupolar value
        line.vars['klse10602'] = k2_value
        twiss = line.twiss() # check stability in twiss command
        
        return line
    
    
    def _print_multipolar_elements_in_line(self, line, order=1):
        """
        Print all quadrupolar elements for a given order (default 1, i.e. quadrupole)
        
        Parameters:
        -----------
        line : xtrack.line
            xtrack line object to search through
        order : int
            multipolar order to print
        
        Returns:
        --------
        None
        """
        
        # Print all quadrupolar components present
        my_dict = line.to_dict()
        d =  my_dict["elements"]
        for key, value in d.items():
            if value['__class__'] == 'Multipole' and value['_order'] == order:
                print('{}: knl = {}'.format(key, value['knl']))