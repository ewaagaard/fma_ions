"""
Tester script to generate SPS sequence with different tunes
"""
import fma_ions
import xtrack as xt

# First test qh_setvalue from MADX
sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=20, num_turns=100)
turns, Qx, Qy = sps_ripple.run_simple_ripple()