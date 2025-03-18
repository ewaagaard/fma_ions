"""
Simple FMA analysis on-momentum for SPS Pb
"""
import fma_ions
import numpy as np
output_dir = './'

# Tracking on GPU context
fma_sps = fma_ions.FMA(output_folder=output_dir, z0=0., n_linear=5)
fma_sps.run_SPS()