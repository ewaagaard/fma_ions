"""
Test tracking with kinetic IBS kicks vs Nagaitsev formalism, for SPS set-up
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=5000, num_turns=20_000, turn_print_interval=10)

# Run with ideal lattice, increase bunch intensity
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = 2.46e8 * 10 # multiply intensity by factor 10
beamParams.sigma_z = 0.12  # decrease bunch lengths to avoid spills
df_kick, df_analytical = sps.run_analytical_vs_kinetic_emittance_evolution(return_data=True, beamParams=beamParams,
                                                                           extra_plot_string='_higher_Nb',
                                                                           show_plot=True)

