"""
Tester script to generate SPS sequence with different tunes
"""
import fma_ions

sps_ripple = fma_ions.Tune_Ripple_SPS()
madx = sps_ripple.find_k_from_q_setvalue()
