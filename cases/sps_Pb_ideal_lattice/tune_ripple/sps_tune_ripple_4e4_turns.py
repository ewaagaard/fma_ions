"""
Tester script to generate SPS sequence with different tunes
"""
import fma_ions
import xtrack as xt

# select resonance and action range we are interested in
test_only_4Qx = True
action_limits=[1.5e-6, 1.7e-6]

if test_only_4Qx:
    # First test qh_setvalue from MADX
    sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=1000, num_turns=40_000, n_linear=50, r_min=2.9, n_sigma=3.1, output_folder='output_4Qx')
    sps_ripple.run_ripple_and_analysis(load_tbt_data=True, install_SC_on_line=True, action_limits=action_limits,
                                       also_show_plot=True)
else:
    # First test qh_setvalue from MADX
    sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=1000, num_turns=40_000, n_linear=50, output_folder='output_tune_ripple')
    sps_ripple.run_ripple_and_analysis(load_tbt_data=True, install_SC_on_line=True, action_limits=action_limits,
                                   also_show_plot=True)
