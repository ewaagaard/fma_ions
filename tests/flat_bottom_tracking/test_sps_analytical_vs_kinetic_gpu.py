"""
Test tracking with kinetic IBS kicks vs Nagaitsev formalism, for SPS set-up - with GPU for longer
(Qx, Qy) = (26.30, 26.19)
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=5000, num_turns=10_000, turn_print_interval=10)

# First run with ideal lattice, then with beta-beat
sps.run_analytical_vs_kinetic_emittance_evolution(Qy_frac=19)
sps.run_analytical_vs_kinetic_emittance_evolution(Qy_frac=19, beta_beat=0.1, add_non_linear_magnet_errors=True)
