"""
Test tracking with kinetic IBS kicks vs Nagaitsev formalism, for SPS set-up - with GPU for longer
(Qx, Qy) = (26.30, 26.19) - ideal lattice
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=5000, num_turns=10_000, turn_print_interval=100)

# First run with ideal lattice, then with beta-beat
sps.run_analytical_vs_kinetic_emittance_evolution(Qy_frac=19, which_context='gpu', ibs_step=300, extra_plot_string='_ideal_lattice')
