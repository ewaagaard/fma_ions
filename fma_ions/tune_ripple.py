"""
Main class containing the tune ripple simulator for SPS
- inspired by xtrack example: https://xsuite.readthedocs.io/en/latest/fast_lattice_changes.html
"""
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
import NAFFlib

import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo

from .sequence_classes_sps import SPS_sequence_maker, BeamParameters_SPS
from .fma_ions import FMA

sequence_path = Path(__file__).resolve().parent.joinpath('../data/sps_sequences').absolute()

@dataclass
class Tune_Ripple_SPS:
    """
    Class to simulate tune ripple by varying quadrupole strengths
    
    Parameters:
    -----------
    Qy_fractional - fractional vertical tune. "19"" means fractional tune Qy = 0.19
    beta_beat - relative beta beat, i.e. relative difference between max beta function and max original beta function
    use_symmetric_lattice - flag to use symmetric lattice without QFA and QDA
    n_linear - default number of points if uniform linear grid for normalized X and Y are used
    r_min - minimum radial distance from beam center to generate particles, to avoid zero amplitude oscillations for FMA
    n_sigma - max number of beam sizes sigma to generate particles
    output_folder - where to save data
    """
    Qy_frac: float = 25
    beta_beat: float = None
    use_symmetric_lattice = True
    ripple_period: int = 2000 
    num_turns: int = 10000
    n_linear: int = 100
    r_min: float = 0.1
    n_sigma: float = 10.0
    output_folder: str = 'output_tune_ripple'
    
    
    def _get_initial_normalized_coord_at_start(self, make_single_Jy_trace=True, y_norm0=0.05):
        """
        Return normalized coordinates of particle object, conforming class parameters
        Stores normalized x and y internally
        
        Parameters
        ----------
        make_single_Jy_trace - flag to create single trace with unique vertical action
        Jy, with varying action Jx. "Trace" instead of "grid", if uniform beam is used
        y_norm0 - starting normalized Y coordinate for the single Jy trace ):
    
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
        
    
    def load_SPS_line_with_deferred_madx_expressions(self, use_symmetric_lattice=True, Qy_frac=25):
        """
        Loads xtrack Pb sequence file with deferred expressions to regulate QD and QF strengths
        or generate from MADX if does not exist
        
        Parameters:
        -----------
        use_symmetric_lattice - flag to use symmetric lattice without QFA and QDA
        Qy_frac - fractional vertical tune. "19"" means fractional tune Qy = 0.19
        
        Returns:
        -------
        xtrack line
        """
        # Try loading existing json file, otherwise create new from MADX
        if use_symmetric_lattice:
            fname = '{}/qy_dot{}/SPS_2021_Pb_symmetric_deferred_exp.json'.format(sequence_path, Qy_frac)
        else:
            fname = '{}/qy_dot{}/SPS_2021_Pb_nominal_deferred_exp.json'.format(sequence_path, Qy_frac)
        
        try: 
            line = xt.Line.from_json(fname)
        except FileNotFoundError:
            print('\nSPS sequence file {} not found - generating new!\n'.format(fname))
            sps = SPS_sequence_maker()
            madx = sps.load_simple_madx_seq(use_symmetric_lattice, Qy_frac=25)
            madx.use(sequence="sps")
    
            # Convert to line
            line = xt.Line.from_madx_sequence(madx.sequence['sps'], deferred_expressions=True)
            m_in_eV, p_inj_SPS = sps.generate_SPS_beam()
            
            line.particle_ref = xp.Particles(
                    p0c = p_inj_SPS,
                    q0 = sps.Q_SPS,
                    mass0 = m_in_eV)
            line.build_tracker()
            
            with open(fname, 'w') as fid:
                json.dump(line.to_dict(), fid, cls=xo.JEncoder)
            
        twiss = line.twiss()
        
        print('\nGenerated SPS Pb beam with gamma = {:.3f}, Qx = {:.3f}, Qy = {:.3f}\n'.format(line.particle_ref.gamma0[0],
                                                                                              twiss['qx'],
                                                                                              twiss['qy']))
        return line, twiss
    
    
    def find_k_from_q_setvalue(self, dq=0.05, plane='X'):
        """
        For desired tune amplitude modulation dQx or dQy, find corresponding change in quadrupole strengths
        'qh_setvalue' and 'qv_setvalue' are linear knobs that regulate QD and QF strength
        - https://gitlab.cern.ch/acc-models/acc-models-sps/-/blame/2021/toolkit/macro.madx?ref_type=heads#L81
        
        Parameters:
        -----------
        dq - change in tune amplitude, e.g. 0.05
        period - oscillation period of tune in number of turns
        plane - 'X' or 'Y' (default is 'X')
        """
        # Load MADX line of SPS and define quad knobs
        sps_seq = SPS_sequence_maker()
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
        
      
    def find_k_from_xtrack_matching(self, dq=0.05, nr_matches=10, use_symmetric_lattice=True, plane='X', Qy_frac=25):
        """
        Find desired tune amplitude modulation dQx or dQy by matching the global
        variable kqf and kqd
        
        Parameters:
        -----------
        dq -  amplitude of oscillations to chosen plane
        nr_matches - interpolation resolution, number of times to match Qx and Qy with propoer kqf and kqd
        use_symmetric_lattice - flag to use symmetric lattice without QFA and QDA
        plane - 'X' or 'Y'
        Qy_frac - fractional vertical tune. "19"" means fractional tune Qy = 0.19
        
        Returns:
        -------
        kqf_vals - numpy array of oscillating kqf strenghts to modulate Qx according to dq (if chosen plane)
        kqd_vals - numpy array of oscillating kqf strenghts to modulate Qy according to dq (if chosen plane)
        turns - numpy array of turns
        """
        # Load Xsuite line with deferred expressions from MADx
        line, twiss = self.load_SPS_line_with_deferred_madx_expressions(use_symmetric_lattice=use_symmetric_lattice,
                                                                        Qy_frac=Qy_frac)
        
        # Empty arrays of quadrupolar strenghts:
        kqfs = np.zeros(nr_matches)
        kqds = np.zeros(nr_matches)
        Qx_vals = np.zeros(nr_matches)
        Qy_vals = np.zeros(nr_matches)
    
        # Investigate linear dependence
        if plane == 'X':
            Qx_target = np.linspace(np.round(twiss['qx'], 2), np.round(twiss['qx'], 2) + dq, num=nr_matches)
            
            for i, Q in enumerate(Qx_target): 
            
                print('\nMatching Qx = {}'.format(Q))
                
                # Match tunes to assigned values
                line.match(
                    vary=[
                        xt.Vary('kqf', step=1e-8),
                        xt.Vary('kqd', step=1e-8),
                    ],
                    targets = [
                        xt.Target('qx', Q, tol=1e-7),
                        xt.Target('qy', twiss['qy'], tol=1e-7)])
                
                twiss = line.twiss()
                
                # Add first values 
                kqfs[i] = line.vars['kqf']._value
                kqds[i] = line.vars['kqd']._value
                Qx_vals[i] = twiss['qx']
                Qy_vals[i] = twiss['qy']
                
                
            # Plot tune evolution over time
            fig, ax = plt.subplots(1, 2, figsize=(12,6))
            ax[0].plot(Qx_vals, kqfs, '-', color='blue')
            ax[1].plot(Qx_vals, kqds, '-', color='red')
            ax[0].set_xlabel('$Q_{x}$')
            ax[0].set_ylabel('kqf')
            ax[1].set_xlabel('$Q_{x}$')
            ax[1].set_ylabel('kqd')
            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.show()
            
        elif plane == 'Y':
            Qy_target = np.linspace(np.round(twiss['qy'], 2), np.round(twiss['qy'], 2) + dq, num=nr_matches)
            
            for i, Q in enumerate(Qy_target): 
            
                print('\nMatching Qx = {}'.format(Q))
                
                # Match tunes to assigned values
                line.match(
                    vary=[
                        xt.Vary('kqf', step=1e-8),
                        xt.Vary('kqd', step=1e-8),
                    ],
                    targets = [
                        xt.Target('qx', twiss['qx'], tol=1e-5),
                        xt.Target('qy', Q, tol=1e-5)])
                
                twiss = line.twiss()
                    
                # Add first values 
                kqfs[i] = line.vars['kqf']._value
                kqds[i] = line.vars['kqd']._value
                Qx_vals[i] = twiss['qx']
                Qy_vals[i] = twiss['qy']
                
                # Plot tune evolution over time
                fig, ax = plt.subplots(1, 2, figsize=(12,6))
                ax[0].plot(Qy_vals, kqfs, '-', color='blue')
                ax[1].plot(Qy_vals, kqds, '-', color='red')
                ax[0].set_xlabel('$Q_{y}$')
                ax[0].set_ylabel('kqf')
                ax[1].set_xlabel('$Q_{y}$')
                ax[1].set_ylabel('kqd')
                fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                plt.show()
        else:
            raise ValueError('Plane not valid - X or Y!')
        
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
        k_val_path = '{}/qy_dot{}/k_knobs'.format(sequence_path, self.Qy_frac)
        os.makedirs(k_val_path, exist_ok=True)
        
        with open("{}/k_vals_{}{}.json".format(k_val_path, plane, sym_string), "w") as fp:
            json.dump(k_dict , fp) 
        
        return kqf_vals, kqd_vals, turns
        
        
    def load_k_from_xtrack_matching(self, dq=0.05, use_symmetric_lattice=True, plane='X'):
        """
        Parameters:
        -----------
        dq -  amplitude of oscillations to chosen plane
        nr_matches - interpolation resolution, number of times to match Qx and Qy with propoer kqf and kqd
        use_symmetric_lattice - flag to use symmetric lattice without QFA and QDA
        plane - 'X' or 'Y'
        
        Returns:
        -------
        kqf_vals - numpy array of oscillating kqf strenghts to modulate Qx according to dq (if chosen plane)
        kqd_vals - numpy array of oscillating kqf strenghts to modulate Qy according to dq (if chosen plane)
        turns - numpy array of turns
        """
        # Trying loading knobs if exist already
        try:
            sym_string = '_symmetric_lattice' if use_symmetric_lattice else '_nominal_lattice'
            k_val_path = '{}/qy_dot{}/k_knobs'.format(sequence_path, self.Qy_frac)
            
            with open("{}/k_vals_{}{}.json".format(k_val_path, plane, sym_string), "r") as fp:
                k_dict = json.load(fp) 
            print('Loaded k_strength json file\n')
                
            # Find amplitudes
            amp_kqf = k_dict['kqf'][-1] - k_dict['kqf'][0]
            amp_kqd = k_dict['kqd'][-1] - k_dict['kqd'][0]
            
            # Create arrays of quadrupole strengths to iterate over
            turns = np.arange(1, self.num_turns+1)
            kqf_vals = k_dict['kqf'][0] + amp_kqf * np.sin(2 * np.pi * turns / self.ripple_period)
            kqd_vals = k_dict['kqd'][0] + amp_kqd * np.sin(2 * np.pi * turns / self.ripple_period)
            
        except FileNotFoundError:
            print('Did not find k_strength json file - creating new!\n')
            
            kqf_vals, kqd_vals, turns = self.find_k_from_xtrack_matching(dq, use_symmetric_lattice=use_symmetric_lattice)
    
        return kqf_vals, kqd_vals, turns
    
    
    def run_simple_ripple_with_twiss(self, dq=0.05, plane='X', use_xtrack_matching=True):
        """
        Run SPS standard tune ripple, with twiss command every turn to check tunes
            
        Parameters:
        -----------
        dq -  amplitude of oscillations to chosen plane
        plane - 'X' or 'Y'
        use_xtrack_matching - flag to use xtrack or MADX qh_setvalue for the matching
        """
        
        # Get SPS Pb line with deferred expressions
        line, twiss = self.load_SPS_line_with_deferred_madx_expressions()
        if use_xtrack_matching:
            kqf_vals, kqd_vals, turns = self.load_k_from_xtrack_matching(dq=dq, plane=plane)
        else:   
            kqf_vals, kqd_vals, turns = self.find_k_from_q_setvalue(dq=dq, plane=plane)
        
        # Generate particles
        fma_sps = FMA(n_linear=self.n_linear, r_min=self.r_min, n_sigma=self.n_sigma)
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
                   use_symmetric_lattice=True, 
                   Qy_frac=25
                   ):
        """
        Test SPS tune ripple during tracking
            
        Parameters:
        -----------
        dq - amplitude of oscillations in chosen plane
        plane - 'X' or 'Y'
        load_tbt_data - flag to load data if already tracked
        save_tbt_data - flag to save tracking data
        make_single_Jy_trace - flag to create single trace with unique vertical action
        use_symmetric_lattice - flag to use symmetric lattice without QFA and QDA
        Qy_frac - fractional vertical tune. "19"" means fractional tune Qy = 0.19
        
        Returns:
        --------
        x, y, px, py - numpy arrays with turn-by-turn data
        """
        # Get SPS Pb line with deferred expressions
        line, twiss = self.load_SPS_line_with_deferred_madx_expressions(use_symmetric_lattice=use_symmetric_lattice, 
                                                                        Qy_frac=Qy_frac)
        kqf_vals, kqd_vals, turns = self.load_k_from_xtrack_matching(dq=dq, plane=plane)

        # Generate particles
        fma_sps = FMA(n_linear=self.n_linear, r_min=self.r_min, n_sigma=self.n_sigma)
        particles = fma_sps.generate_particles(line, BeamParameters_SPS, make_single_Jy_trace=make_single_Jy_trace)
        
        # Empty array for turns
        x = np.zeros([len(particles.x), self.num_turns]) 
        y = np.zeros([len(particles.y), self.num_turns])
        px = np.zeros([len(particles.px), self.num_turns]) 
        py = np.zeros([len(particles.py), self.num_turns])
        
        # Track the particles and return turn-by-turn coordinates
        for ii in range(self.num_turns):
            if ii % 20 == 0: print(f'Turn {ii} of {self.num_turns}')
            
            x[:, ii] = particles.x
            y[:, ii] = particles.y
            px[:, ii] = particles.px
            py[:, ii] = particles.py
        
            # Change the strength of the quads
            line.vars['kqf'] = kqf_vals[ii]
            line.vars['kqd'] = kqd_vals[ii]
        
            # Track one turn
            line.track(particles)
        
        # Set particle trajectories of dead particles that got lost in tracking
        self._kill_ind = particles.state < 1
        self._kill_ind_exists = True
        
        if save_tbt_data:
            os.makedirs(self.output_folder, exist_ok=True)
            np.save('{}/x.npy'.format(self.output_folder), x)
            np.save('{}/y.npy'.format(self.output_folder), y)
            np.save('{}/px.npy'.format(self.output_folder), px)
            np.save('{}/py.npy'.format(self.output_folder), py)
            np.save('{}/state.npy'.format(self.output_folder), self._kill_ind)
            print('Saved tracking data.')
    
        return x, y, px, py
    
    
    def load_tracking_data(self):
        """Loads numpy data if tracking has already been made"""
        try:            
            x=np.load('{}/x.npy'.format(self.output_folder))
            y=np.load('{}/y.npy'.format(self.output_folder))
            px=np.load('{}/px.npy'.format(self.output_folder))
            py=np.load('{}/py.npy'.format(self.output_folder))
            return x, y, px, py

        except FileNotFoundError:
            raise FileNotFoundError('Tracking data does not exist - set correct path or generate the data!')
    
    
    def load_tune_data(self):
        """Loads numpy data of tunes if FMA has already been done"""
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
        x_tbt_data, y_tbt_data - numpy arrays of turn-by-turn data for particles 
        Qmin - if desired, filter out some lower frequencies
        k - length of period to evaluate tune over, "tune period"
        save_tune_data - flag to save tune data
        
        Returns: 
        --------
        """
        # Iterate over particles to find tune
        for i_part in range(len(x_tbt_data)):
            
            if i_part % 2 == 0:
                print('NAFF algorithm of particle {}'.format(i_part))
            
            # Calculate the index "window" from which tunes are extracted 
            L = len(x_tbt_data[0]) # number of turns
            
            Qx = np.zeros([len(x_tbt_data), L - k + 1])
            Qy = np.zeros([len(x_tbt_data), L - k + 1])
            
            # Iterate over subwindows of length k to find tune
            for i in range(L - k + 1):
                
                if i_part % 2 == 0 and i % 100 == 0:
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
                   use_symmetric_lattice=True, 
                   Qy_frac=25):
        """
        Run SPS tune ripple wtih tracking and generate phase space plots
            
        Parameters:
        -----------
        dq - amplitude of oscillations in chosen plane
        plane - 'X' or 'Y'
        load_tbt_data - flag to load data if already tracked
        make_single_Jy_trace - flag to create single trace with unique vertical action
        use_symmetric_lattice - flag to use symmetric lattice without QFA and QDA
        Qy_frac - fractional vertical tune. "19"" means fractional tune Qy = 0.19
        
        Returns:
        --------
        """
        if load_tbt_data:
            x, y, px, py = self.load_tracking_data()
        else:
            x, y, px, py = self.run_ripple(dq=dq, plane=plane, make_single_Jy_trace=make_single_Jy_trace)
        
        # Load relevant SPS line and twiss
        self._get_initial_normalized_coord_at_start() # loads normalized coord of starting distribution
        line, twiss = self.load_SPS_line_with_deferred_madx_expressions(use_symmetric_lattice=use_symmetric_lattice,
                                                                        Qy_frac=Qy_frac)
        
        # Try to load tune-data
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
        
        # Calculate action for each particle
        Jx = X**2 + PX **2
        Jy = Y**2 + PY **2

        # Action evolution over time
        fig, ax = plt.subplots(1, 1, figsize=(12,6))
        ax.plot(turns, Jx[-1], '-', color='blue')
        ax.set_xlabel('Turns')
        ax.set_ylabel('$J_{x}$')
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()
    
    
    def print_quadrupolar_elements_in_line(self, line):
        """Print all quadrupolar elements"""
        
        # Print all quadrupolar components present
        my_dict = line.to_dict()
        d =  my_dict["elements"]
        for key, value in d.items():
            if value['__class__'] == 'Multipole' and value['_order'] == 1:
                print('{}: knl = {}'.format(key, value['knl']))