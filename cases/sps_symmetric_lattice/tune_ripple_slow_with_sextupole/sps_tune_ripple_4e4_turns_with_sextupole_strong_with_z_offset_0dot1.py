"""
Tester script to generate SPS sequence with different tunes
- set additional sextupolar value to one LSE
"""
import fma_ions
import xtrack as xt

# First test qh_setvalue from MADX
sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=10_000, num_turns=40_000, n_linear=30, z0=0.1,
                                      output_folder='output_tune_ripple_k2_strong_with_z_offset_0dot1')
sps_ripple.run_ripple_and_analysis(load_tbt_data=True, install_SC_on_line=True, sextupolar_value_to_add=0.5, plot_random_colors=True,
                               also_show_plot=True, action_in_logscale=True)