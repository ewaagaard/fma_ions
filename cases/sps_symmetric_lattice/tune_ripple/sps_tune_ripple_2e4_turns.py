"""
Tester script to generate SPS sequence with different tunes
- select to generate subset of particles on specific resonance
"""
import fma_ions

sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=10_000, num_turns=20_000, n_linear=30, output_folder='output_tune_ripple')
sps_ripple.run_ripple_and_analysis(load_tbt_data=True, install_SC_on_line=True, plot_random_colors=True,
                               also_show_plot=True, phase_sweep_up_to_turn=20_000, phase_space_sweep_interval=200)


"""
# select resonance and action range we are interested in
test_only_8Qx = True
action_limits=[1.5e-6, 1.7e-6]

if test_only_8Qx:
    # First test qh_setvalue from MADX
    sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=1000, num_turns=40_000, n_linear=50, r_min=2.9, n_sigma=3.1, output_folder='output_8Qx')
    sps_ripple.run_ripple_and_analysis(load_tbt_data=True, install_SC_on_line=True, action_limits=action_limits,
                                       also_show_plot=False)
else:
    # First test qh_setvalue from MADX
    sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=1000, num_turns=40_000, n_linear=50, output_folder='output_tune_ripple')
    sps_ripple.run_ripple_and_analysis(load_tbt_data=True, install_SC_on_line=True, action_limits=action_limits,
                                   also_show_plot=True)
"""