"""
Test to check if particles fall out of bucket for SPS Pb ions
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=5000, num_turns=100, turn_print_interval=10)

beamParams1 = fma_ions.BeamParameters_SPS()
beamParams1.sigma_z = 0.225 

beamParams2 = fma_ions.BeamParameters_SPS()
beamParams2.sigma_z = 0.19 

# First test 
sps.run_analytical_vs_kinetic_emittance_evolution(beamParams=beamParams1, extra_plot_string='_sigmaZ_0dot225')
sps.run_analytical_vs_kinetic_emittance_evolution(beamParams=beamParams2, extra_plot_string='_sigmaZ_0dot19')