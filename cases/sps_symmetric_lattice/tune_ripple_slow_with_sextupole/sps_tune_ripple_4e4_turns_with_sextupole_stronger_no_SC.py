"""
Tester script to generate SPS sequence with different tunes
- set additional sextupolar value to one LSE, now 0.5 instead of 0.1
- no space charge active
"""
import fma_ions
import xtrack as xt

# First test qh_setvalue from MADX
sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=10_000, num_turns=20_000, n_linear=30, output_folder='output_tune_ripple_k2_0dot5_no_SC')
sps_ripple.run_ripple_and_analysis(load_tbt_data=False, install_SC_on_line=False, sextupolar_value_to_add=0.5, plot_random_colors=True,
                               also_show_plot=True, action_in_logscale=True, phase_sweep_up_to_turn=5000, phase_space_sweep_interval=50)
