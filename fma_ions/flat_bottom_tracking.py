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

@dataclass
class SPS_Flat_Bottom_Tracker:
    """
    Container to track xp.Particles at SPS flat bottom and store beam parameter results
    """
    num_part: int = 5000
    num_turns: int = 1000
    Qy_frac: int = 25

    def track_SPS(self, sc_mode='frozen', 
                  save_tbt_data=True, 
                  context='gpu',
                  add_non_linear_magnet_errors=False, 
                  add_aperture=True,
                  beta_beat=None, 
                  beamParams=None,
                  install_SC_on_line=True, 
                  SC_mode='frozen'
                  ):
        """
        save_tbt: bool
            whether to save turn-by-turn data from tracking
        context : str
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

        Returns:
        -------
        tbt : data class with numpy.ndarrays
        """
        # If specific beam parameters are not provided, load default SPS beam parameters
        if beamParams is None:
            beamParams = BeamParameters_SPS

        # Select relevant context
        if context=='gpu':
            context = xo.ContextCupy()
        elif context=='cpu':
            context = xo.ContextCpu()
        else:
            raise ValueError('Context is either "gpu" or "cpu"')

        # Get SPS Pb line - with beta-beat, aperture and non-linear magnet errors if desired
        sps = SPS_sequence_maker()
        line, twiss = sps.load_SPS_line_and_twiss(Qy_frac=self.Qy_frac, add_aperture=add_aperture, beta_beat=None,
                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors)
        

    
        # Install SC and build tracker
        if install_SC_on_line:
            fma_sps = FMA()
            line = fma_sps.install_SC_and_get_line(line, beamParams, mode=SC_mode, optimize_for_tracking=False, context=context)
            print('Installed space charge on line\n')
        else:
            line.discard_tracker()
            line.build_tracker(_context=context)


        


        # Initialize the dataclasses
        tbt = Records.init_zeroes(self.num_turns)

        # Store the initial values
        tbt.update_at_turn(0, particles, twiss)

        print('\nStarting tracking...')
        i = 0
        for turn in range(self.num_turns):
            if i % 100 == 0:
                print('Tracking turn {}'.format(i))
        
            # ----- Track and update records for tracked particles ----- #
            line.track(particles, num_turns=1)
            tbt.update_at_turn(turn, particles, twiss)
            i += 1

        # Save data
        # Analyze data / plot
        
