"""
Generate SPS Pb ion FMA plot
"""
import fma_ions

fma_sps = fma_ions.FMA()
fma_sps.run_SPS()


#x, y = fma_sps.load_tracking_data()
#d, Qx, Qy = fma_sps.run_FMA(x, y)

