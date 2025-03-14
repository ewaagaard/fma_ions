"""
Example script for single particle 10k turns with kick
"""
import fma_ions
import numpy as np
output_dir = './'

n_turns = 70_000
sps_plot = fma_ions.SPS_Plotting()
sps_kick = fma_ions.SPS_Kick_Plotter()

# Transfer function factors
a_50 = 1.0 #1.7170
a_150 = 0.5098
a_300 = 0.2360
a_600 = 0.1095

# Desired ripple frequencies and amplitudes
ripple_freqs = np.array([50.0, 150.0, 300.0, 600.0])
kqf_amplitudes = np.array([1.6384433351717334e-08*a_50, 2.1158318710898557e-07*a_150, 3.2779826135772383e-07*a_300, 4.7273849059164697e-07*a_600])
kqd_amplitudes = np.array([1.6331206486868624e-08*a_50, 2.108958328708343e-07*a_150, 3.2673336803004776e-07*a_300, 4.7120274094403095e-07*a_600])
kqf_phases = np.array([0.9192671763874849, 0.030176158557178895, 0.5596488397663701, 0.050511945653341016])
kqd_phases = np.array([-2.2223254772020873, -3.1114164950326417, -2.581943813823403, -3.0910807079364635])

try:
    tbt_dict = sps_plot.load_records_dict_from_json()

except FileNotFoundError:
    # Tracking on GPU context
    sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0=26.31, qy0=26.19, dqx0=0.0, dqy0=0.0, num_turns=n_turns, turn_print_interval=500)
    tbt = sps.track_SPS(which_context='cpu', distribution_type='single', install_SC_on_line=False,
                        add_tune_ripple=True, ripple_freqs = ripple_freqs, kqf_amplitudes = kqf_amplitudes, 
                        kqd_amplitudes = kqd_amplitudes, kqf_phases=kqf_phases, kqd_phases=kqd_phases, kick_beam=True)
    tbt.to_json(output_dir)
    tbt_dict = tbt.to_dict()

sps_kick.plot_tbt_data_to_spectrum(tbt_dict, ripple_freqs=ripple_freqs, transfer_function_bounds = [10., 15500.])
