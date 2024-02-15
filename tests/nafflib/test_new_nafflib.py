"""
Tester analyzing the tunes with new NAFFlib
"""
import fma_ions
import xtrack as xt

fma_sps = fma_ions.FMA(num_turns=60, n_linear=25, output_folder='../sps/Output_test')
x, y, _, _ = fma_sps.load_tracking_data()
Qx, Qy, d = fma_sps.run_FMA(x, y)