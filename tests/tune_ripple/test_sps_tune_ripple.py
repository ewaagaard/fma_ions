"""
Tester script to generate SPS sequence with different tunes
"""
import fma_ions
import xtrack as xt

# Test if tunes are correct - first try only in X, then in both planes
sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=20, num_turns=100)
#turns, Qx, Qy = sps_ripple.run_simple_ripple_with_twiss(plane='X')
turns2, Qx2, Qy2 = sps_ripple.run_simple_ripple_with_twiss(plane='both')