"""
Generate SPS O ion FMA plot - ideal lattice, no momentum offset
"""
import fma_ions

fma_sps = fma_ions.FMA(output_folder='output_O_on_momentum_ideal_lattice_Nb_2dot5e9', z0=0., n_linear=200)
fma_sps.run_SPS(load_tbt_data=True, ion_type='O', which_context='gpu')
