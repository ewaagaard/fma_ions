"""
Generate PS Pb ion FMA plot - ideal lattice, with zero momentum offset
"""
import fma_ions

fma_ps = fma_ions.FMA(output_folder='output_Pb_on_momentum_ideal_lattice', z0=0.0, n_linear=200)
fma_ps.run_PS()
