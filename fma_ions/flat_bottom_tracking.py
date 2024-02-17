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


@dataclass
class Records:
    x: np.ndarray
    y: np.ndarray
    nepsilon_x: np.ndarray
    nepsilon_y: np.ndarray
    sigma_delta: np.ndarray
    bunch_length: np.ndarray
    Nb: np.ndarray
    state: np.ndarray

    def update_at_turn(self, turn: int, parts: xp.Particles, twiss: xt.TwissTable):
        """Automatically update the records at given turn from the xpart.Particles."""
        self.x = parts.x[parts.state > 0]
        self.y = parts.y[parts.state > 0]
        self.nepsilon_x[turn] = _geom_epsx(parts, twiss) * parts.beta0[0] * parts.gamma0[0]
        self.nepsilon_y[turn] = _geom_epsy(parts, twiss) * parts.beta0[0] * parts.gamma0[0]
        self.Nb[turn] = parts.weight[0]*len(parts.x[parts.state > 0])
        self.sigma_delta[turn] = _sigma_delta(parts)
        self.bunch_length[turn] = _bunch_length(parts)
        self.state[turn] = parts.state

    @classmethod
    def init_zeroes(cls, n_turns: int) -> Self:  # noqa: F821
        """Initialize the dataclass with arrays of zeroes."""
        return cls(
            x=np.zeros(n_turns, dtype=float),
            y=np.zeros(n_turns, dtype=float),
            nepsilon_x=np.zeros(n_turns, dtype=float),
            nepsilon_y=np.zeros(n_turns, dtype=float),
            Nb=np.zeros(n_turns, dtype=float),
            sigma_delta=np.zeros(n_turns, dtype=float),
            bunch_length=np.zeros(n_turns, dtype=float),
            state=np.zeros(n_turns, dtype=float)
        )
#############################################################################


class SPS_Flat_Bottom_Tracker:
    """
    Container to track particles 
    """
    num_part:
    num_turns:

    def track_SPS(self, sc_mode='frozen', save_tbt_data=True, context='gpu'):
        """
        save_tbt: bool
            whether to save turn-by-turn data from tracking
        context : str
            'gpu' or 'cpu'

        Returns:
        -------
        tbt : data class with numpy.ndarrays
        """
        # Select relevant context
        if context=='gpu':
            context = xo.ContextCupy()
        elif context=='cpu':
            context = xo.ContextCpu()
        else:
            raise ValueError('Context is either "gpu" or "cpu"')

        # Get SPS Pb line with deferred expressions
        sps = SPS_sequence_maker()
        line, twiss = sps.load_SPS_line_and_twiss(Qy_frac=Qy_frac, add_aperture=add_aperture, beta_beat=None,
                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors)
        line.discard_tracker()
        line.build_tracker(_context=context)
        

        # Fix SC, beta-beat and non-linear magnet errors

        # Install SC, track particles and observe tune diffusion
        
        # Add beta-beat if desired 
        if beta_beat is not None:
            sps_seq = SPS_sequence_maker()
            line = sps_seq.generate_xsuite_seq_with_beta_beat(beta_beat=beta_beat, line=line, plane=plane_beta_beat)

        if install_SC_on_line:
            fma_sps = FMA()
            line = fma_sps.install_SC_and_get_line(line, BeamParameters_SPS(), optimize_for_tracking=False)
            print('Installed space charge on line\n')
        kqf_vals, kqd_vals, turns = self.load_k_from_xtrack_matching(dq=dq, plane=plane)

        line = self.install_SC_and_get_line(line0, beamParams)
        


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
        
