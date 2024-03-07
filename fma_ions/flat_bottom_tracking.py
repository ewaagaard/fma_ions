"""
Class container for methods to track xpart particle objects at flat bottom
- for SPS
- choose context (GPU, CPU) and additional effects: SC, IBS, tune ripples
"""
from dataclasses import dataclass
import numpy as np
import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo

from .sequence_classes_ps import PS_sequence_maker, BeamParameters_PS
from .sequence_classes_sps import SPS_sequence_maker, BeamParameters_SPS
from .fma_ions import FMA
from .helpers import Records, _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta

from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS

import os
import matplotlib.pyplot as plt

@dataclass
class SPS_Flat_Bottom_Tracker:
    """
    Container to track xp.Particles at SPS flat bottom and store beam parameter results
    """
    num_part: int = 5000
    num_turns: int = 1000
    Qy_frac: int = 25
    _output_folder : str = "output"
    turn_print_interval : int = 100

    def generate_particles(self, line: xt.Line, context : xo.context, use_Gaussian_distribution=True, beamParams=None
                           ) -> xp.Particles:
        """
        Generate xp.Particles object: matched Gaussian or other types (to be implemented)
        """
        if beamParams is None:
            beamParams = BeamParameters_SPS

        if use_Gaussian_distribution:
            particles = xp.generate_matched_gaussian_bunch(_context=context,
                num_particles=self.num_part, 
                total_intensity_particles=beamParams.Nb,
                nemitt_x=beamParams.exn, 
                nemitt_y=beamParams.eyn, 
                sigma_z= beamParams.sigma_z,
                particle_ref=line.particle_ref, 
                line=line)
            
        return particles


    def track_SPS(self, 
                  save_tbt_data=True, 
                  which_context='gpu',
                  add_non_linear_magnet_errors=False, 
                  add_aperture=True,
                  beta_beat=None, 
                  beamParams=None,
                  install_SC_on_line=True, 
                  SC_mode='frozen',
                  use_Gaussian_distribution=True,
                  apply_kinetic_IBS_kicks=False,
                  harmonic_nb = 4653,
                  ibs_step = 50
                  ):
        """
        save_tbt: bool
            whether to save turn-by-turn data from tracking
        which_context : str
            'gpu' or 'cpu'
        Qy_frac : int
            fractional part of vertical tune
        add_non_linear_magnet_errors : bool
            whether to add line with non-linear chromatic errors
        add_aperture : bool
            whether to include aperture for SPS
        beta_beat : float
            relative beta beat, i.e. relative difference between max beta function and max original beta function
        beamParams : dataclass
            container of exn, eyn, Nb and sigma_z. Default 'None' will load nominal SPS beam parameters 
        install_SC_on_line : bool
            whether to install space charge
        SC_mode : str
            type of space charge - 'frozen' (recommended), 'quasi-frozen' or 'PIC'
        use_Gaussian_distribution : bool
            whether to use Gaussian particle distribution for tracking
        add_kinetic_IBS_kicks : bool
            whether to apply kinetic kicks from xibs 
        harmonic_nb : int
            harmonic used for SPS RF system
        ibs_step : int
            turn interval at which to recalculate IBS growth rates
        """
        # If specific beam parameters are not provided, load default SPS beam parameters
        if beamParams is None:
            beamParams = BeamParameters_SPS

        # Select relevant context
        if which_context=='gpu':
            context = xo.ContextCupy()
        elif which_context=='cpu':
            context = xo.ContextCpu()
        else:
            raise ValueError('Context is either "gpu" or "cpu"')

        # Get SPS Pb line - with beta-beat, aperture and non-linear magnet errors if desired
        sps = SPS_sequence_maker()
        line0, twiss = sps.load_xsuite_line_and_twiss(Qy_frac=self.Qy_frac, add_aperture=add_aperture, beta_beat=beta_beat,
                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors)
        
        # Add longitudinal limit rectangle - to kill particles that fall out of bucket
        bucket_length = line0.get_length()/harmonic_nb
        line0.unfreeze() # if you had already build the tracker
        line0.append_element(element=xt.LongitudinalLimitRect(min_zeta=-bucket_length/2, max_zeta=bucket_length/2), name='long_limit')
        
        # Install SC and build tracker
        if install_SC_on_line:
            fma_sps = FMA()
            line = fma_sps.install_SC_and_get_line(line0, beamParams, mode=SC_mode, optimize_for_tracking=True, context=context)
            print('Installed space charge on line\n')
        else:
            line = line0.copy()
            line.discard_tracker()
            line.build_tracker(_context=context)

        # Generate particles object to track
        particles = self.generate_particles(line=line, context=context, use_Gaussian_distribution=use_Gaussian_distribution,
                                            beamParams=beamParams)

        # Initialize the dataclasses and store the initial values
        tbt = Records.init_zeroes(self.num_turns)
        tbt.update_at_turn(0, particles, twiss)

        ######### IBS kinetic kicks #########
        if apply_kinetic_IBS_kicks:
            beamparams = BeamParameters.from_line(line, n_part=beamParams.Nb)
            opticsparams = OpticsParameters.from_line(line)
            IBS = KineticKickIBS(beamparams, opticsparams)
            kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
            print(kinetic_kick_coefficients)

        print('\nStarting tracking...')
        i = 0
        for turn in range(self.num_turns):
            
            if turn % self.turn_print_interval == 0:
                print('Tracking turn {}'.format(i))            

            ########## IBS -> Potentially re-compute the ellitest_parts integrals and IBS growth rates #########
            if apply_kinetic_IBS_kicks and ((turn % ibs_step == 0) or (turn == 1)):
                print(
                    "=" * 60 + "\n",
                    f"Turn {turn:d}: re-computing growth rates and kick coefficients\n",
                    "=" * 60,
                )
                # We compute from values at the previous turn
                kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
                print(kinetic_kick_coefficients)
                
            ########## ----- Apply IBS Kick if desired ----- ##########
            if apply_kinetic_IBS_kicks:
                IBS.apply_ibs_kick(particles)
            
            # ----- Track and update records for tracked particles ----- #
            line.track(particles, num_turns=1)
            tbt.update_at_turn(turn, particles, twiss)
            i += 1

        if save_tbt_data: 
            os.makedirs(self._output_folder, exist_ok=True)
            np.save('{}/nepsilon_x.npy'.format(self._output_folder), tbt.nepsilon_x)
            np.save('{}/nepsilon_y.npy'.format(self._output_folder), tbt.nepsilon_y)
            np.save('{}/sigma_delta.npy'.format(self._output_folder), tbt.sigma_delta)
            np.save('{}/bunch_length.npy'.format(self._output_folder), tbt.bunch_length)
            np.save('{}/Nb.npy'.format(self._output_folder), tbt.Nb)

        self.plot_tracking_data(tbt)


    def load_tbt_data(self) -> Records:
        """
        Loads numpy data if tracking has already been made
        """

        # First initialize empty data class to fill with data
        tbt = Records.init_zeroes(self.num_turns)
        tbt.nepsilon_x = np.load('{}/nepsilon_x.npy'.format(self._output_folder))
        tbt.nepsilon_y = np.load('{}/nepsilon_y.npy'.format(self._output_folder))
        tbt.sigma_delta = np.load('{}/sigma_delta.npy'.format(self._output_folder))
        tbt.bunch_length = np.load('{}/bunch_length.npy'.format(self._output_folder))
        tbt.Nb = np.load('{}/Nb.npy'.format(self._output_folder))
        
        return tbt


    def plot_tracking_data(self, tbt):
        """Generates emittance plots from TBT data class"""

        turns = np.arange(self.num_turns, dtype=int) 

        # Emittances and bunch intensity 
        f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,5))

        ax1.plot(turns, tbt.nepsilon_x * 1e6, alpha=0.7, lw=1.5, label='X')
        ax1.plot(turns, tbt.nepsilon_y * 1e6, lw=1.5, label='Y')
        ax2.plot(turns, tbt.Nb, alpha=0.7, lw=1.5, c='r', label='Bunch intensity')

        ax1.set_ylabel(r'$\varepsilon_{x, y}$ [$\mu$m]')
        ax1.set_xlabel('Turns')
        ax2.set_ylabel(r'$N_{b}$')
        ax2.set_xlabel('Turns')
        ax1.legend()
        f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        # Emittances and bunch intensity 
        f2, (ax12, ax22) = plt.subplots(1, 2, figsize = (8,4))

        ax12.plot(turns, tbt.sigma_delta * 1e3, alpha=0.7, lw=1.5, label='$\sigma_{\delta}$')
        ax22.plot(turns, tbt.bunch_length, alpha=0.7, lw=1.5, label='Bunch intensity')

        ax12.set_ylabel(r'$\sigma_{\delta}$')
        ax12.set_xlabel('Turns')
        ax22.set_ylabel(r'$\sigma_{z}$ [m]')
        ax22.set_xlabel('Turns')    
        f2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        plt.show()


    def load_tbt_data_and_plot(self):
        """Load already tracked data and plot"""
        try:
            tbt = self.load_tbt_data()
            self.plot_tracking_data(tbt)
        except FileNotFoundError:
            raise FileNotFoundError('Tracking data does not exist - set correct path or generate the data!')
        

    def plot_multiple_sets_of_tracking_data(self, tbt_array, string_array):
        """
        If multiple runs with turn-by-turn data has been made, provide list with Records class objects and list
        of explaining string to generate comparative plots of emittances, bunch intensities, etc

        Parameters:
        ----------
        tbt_array : [Records, Records, ...]
            List containing tbt data in Records class format
        string:_array : [str1, str2, ...]
            List containing strings to explain the respective tbt data objects (which parameters were used)
        """