"""
Class container for methods to track xpart particle objects at flat bottom
- for SPS
- choose context (GPU, CPU) and additional effects: SC, IBS, tune ripples
"""
from dataclasses import dataclass

from .sequence_classes_ps import PS_sequence_maker, BeamParameters_PS
from .sequence_classes_sps import SPS_sequence_maker, BeamParameters_SPS
from .fma_ions import FMA

class SPS_Flat_Bottom_Tracker:
    """
    Container to track particles 
    """

    def track_SPS(self, sc_mode='frozen'):

        # Get SPS Pb line with deferred expressions
        sps = SPS_sequence_maker()
        line, twiss = sps.load_SPS_line_and_twiss(Qy_frac=Qy_frac, add_aperture=add_aperture, beta_beat=None,
                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors)
        
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
        particles = self.generate_particles(line, beamParams, make_single_Jy_trace)
        x, y = self.track_particles(particles, line)


        
    def track_particles(self, line, particles):
        # Save emittances, BL, sigma_delta and particle states every turns
        # Follow example from IBS tester