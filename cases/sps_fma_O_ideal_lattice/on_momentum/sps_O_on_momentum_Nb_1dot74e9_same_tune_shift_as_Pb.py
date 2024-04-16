"""
Generate SPS O ion FMA plot - ideal lattice, no momentum offset - 
set Nb to match dQ of present Pb parameters
"""
import fma_ions

beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = 1.74e9

fma_sps = fma_ions.FMA(output_folder='output_O_on_momentum_ideal_lattice_Nb_1dot74e9_same_dQ_as_Pb', z0=0., n_linear=100)
fma_sps.run_SPS(load_tbt_data=False, ion_type='O', which_context='gpu', beamParams=beamParams)
