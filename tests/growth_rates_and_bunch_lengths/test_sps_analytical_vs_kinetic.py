"""
Test tracking with kinetic IBS kicks vs Nagaitsev formalism, for SPS set-up
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=5000, num_turns=20_000, turn_print_interval=10)

# First run with ideal lattice, then with beta-beat
df_kick, df_analytical = sps.run_analytical_vs_kinetic_emittance_evolution(return_data=True, show_plot=True)
