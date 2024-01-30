"""
Tester script to generate SPS sequence with different tunes
- test 10% beta-beat and lse of 0.01
"""
import fma_ions
import xtrack as xt

# First test qh_setvalue from MADX
sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=10_000, num_turns=20_000, n_linear=30, 
                                      output_folder='output_tune_ripple_bb_0dot1_lse_0dot01_nonlinear_errors')
sps_ripple.run_ripple_and_analysis(load_tbt_data=False, install_SC_on_line=True, sextupolar_value_to_add=0.01, plot_random_colors=True,
                               also_show_plot=True, phase_sweep_up_to_turn=20_000, phase_space_sweep_interval=200, beta_beat=0.1, 
                               add_non_linear_magnet_errors=True)