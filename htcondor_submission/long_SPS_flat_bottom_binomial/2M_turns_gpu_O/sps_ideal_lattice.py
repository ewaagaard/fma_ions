"""
SPS ideal lattice with NO space charge - with GPUs for 2M turns
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 2_000_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns)
tbt = sps.track_SPS(ion_type='O', which_context='gpu', distribution_type='binomial', install_SC_on_line=False)
tbt.to_parquet(f'{output_dir}/tbt.parquet')