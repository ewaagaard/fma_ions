"""
Tester script to generate SPS sequence with different tunes
- select to generate subset of particles on specific resonance
"""
import fma_ions

sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=10_000, num_turns=20_000, n_linear=30, output_folder='output_tune_ripple')
sps_ripple.run_ripple_and_analysis(load_tbt_data=False, install_SC_on_line=True, plot_random_colors=True, use_symmetric_lattice=False,
                               also_show_plot=True, phase_sweep_up_to_turn=20_000, phase_space_sweep_interval=200)

