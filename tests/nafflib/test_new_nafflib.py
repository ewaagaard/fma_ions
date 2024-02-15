"""
Tester analyzing the tunes with new NAFFlib
"""
import fma_ions
import xtrack as xt

# Try generated data from on-momentum fma
fma_sps = fma_ions.FMA(num_turns=60, n_linear=25, output_folder='../../cases/sps_Pb_ideal_lattice/first_cases_fma_plots_for_momentum_offsets/output_Pb_on_momentum_ideal_lattice')
fma_sps.run_SPS(load_tbt_data=True, load_tune_data=False)