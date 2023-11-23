"""
Generate SPS Pb ion FMA plot
"""
import fma_ions

fma_sps = fma_ions.FMA()
fma_sps.run_SPS(load_tbt_data=True)
