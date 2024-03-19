"""
Test tracking with kinetic IBS kicks vs Nagaitsev formalism, for SPS set-up
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=2000, num_turns=100, turn_print_interval=10)

# First run with ideal lattice, then with beta-beat
sps.run_analytical_vs_kinetic_emittance_evolution()
sps.run_analytical_vs_kinetic_emittance_evolution(beta_beat=0.1, add_non_linear_magnet_errors=True)
