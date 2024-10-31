"""
Check octupolar k3 components from Twiss with SPS Pb sequence used for FMA 
"""
import fma_ions


# Test Twiss and tune adjustments of SPS 
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()
 
line2 = sps.set_LOE_octupolar_errors(line)
line2.twiss()



