"""
Simple FMA analysis on-momentum for SPS Pb
"""
import fma_ions
output_dir = './'

# Test simple FMA on CPU context
fma = fma_ions.FMA(n_linear=5)
tbt = fma.run_SPS()
tbt.to_json(output_dir)

# Try loading data and plot the resulting diagram
fma_plot = fma_ions.FMA_plotter()
fma_plot.plot_FMA()
