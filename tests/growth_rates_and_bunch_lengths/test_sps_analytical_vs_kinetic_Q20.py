"""
Test tracking with kinetic IBS kicks vs Nagaitsev formalism, for SPS set-up but with Q20 optics
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=10_000, num_turns=2_000, turn_print_interval=10, qx0=20.3, qy0=20.25, proton_optics='q20')

# First run with ideal lattice, then with beta-beat
df_kick, df_analytical = sps.run_analytical_vs_kinetic_emittance_evolution(return_data=True, show_plot=True, distribution_type='binomial')
