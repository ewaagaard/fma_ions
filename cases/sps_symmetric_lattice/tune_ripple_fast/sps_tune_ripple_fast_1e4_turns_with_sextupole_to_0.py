"""
Tester script to generate SPS sequence with different tunes
- fast tune ripple over 1000 turns 
- set additional sextupolar value to one LSE
"""
import fma_ions
import xtrack as xt

# First test qh_setvalue from MADX
sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=1_000, num_turns=10_000, n_linear=30, output_folder='output_tune_ripple_fast_k2_0')
sps_ripple.run_ripple_and_analysis(load_tbt_data=False, install_SC_on_line=True, sextupolar_value_to_add=0.0, plot_random_colors=True,
                               also_show_plot=True, phase_sweep_up_to_turn=10000, phase_space_sweep_interval=200)
