"""
Tester script to generate SPS sequence with different tunes
"""
import fma_ions
import xtrack as xt

# First test qh_setvalue from MADX
sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=10_000, num_turns=200_000, n_linear=500, output_folder='output_tune_ripple_long')
sps_ripple.run_ripple_and_analysis(load_tbt_data=True, install_SC_on_line=True, num_particles_to_plot=50,
                                   also_show_plot=True)