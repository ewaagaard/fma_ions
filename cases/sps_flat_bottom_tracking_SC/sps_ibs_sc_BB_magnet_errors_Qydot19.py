"""
SPS ideal lattice with space charge and IBS, magnet errors, BB of 10% and Qy changed to 26.19 
as during measurements 
"""
import fma_ions

# Update beam parameters to what was observed for Pb beams in SPS on 2023-10-16
beamParams = fma_ions.BeamParameters_SPS
beamParams.Nb = 2.0e8
beamParams.exn = 0.3e-6
beamParams.eyn = 0.3e-6
print(beamParams)

# Test default tracking with space charge and IBS on GPU context 
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=50, output_folder='output_ibs_sc_full_Qy_26dot19')
tbt = sps.track_SPS(which_context='cpu', add_aperture=True, install_SC_on_line=True, apply_kinetic_IBS_kicks=True, ibs_step=100,
                    add_non_linear_magnet_errors=True, beta_beat=0.1, Qy_frac=19, beamParams=beamParams)
