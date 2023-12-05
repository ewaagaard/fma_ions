"""
Generate PS Pb ion FMA plot - ideal lattice, with momentum offset
"""
import fma_ions

fma_ps = fma_ions.FMA(output_folder='output_Pb_off_momentum_ideal_lattice', z0=0.15, n_linear=200)
fma_ps.run_PS()
