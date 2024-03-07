"""
Small tester to ensure that loading a MADX instance into xtrack with deferred expressions and aperture works properly 
"""
import fma_ions
import xtrack
import xpart

sps = fma_ions.SPS_sequence_maker()
#line, twiss = sps.load_SPS_line_with_deferred_madx_expressions(add_aperture=True)

# Also test beta-beat with aperture
line = sps.generate_xsuite_seq_with_beta_beat(beta_beat=0.1, add_aperture=True)