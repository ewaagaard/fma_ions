"""
Example script for single particle 10k turns with kick
"""
import fma_ions
import numpy as np
output_dir = './'

n_turns = 10_000
sps_plot = fma_ions.SPS_Plotting()
sps_kick = fma_ions.SPS_Kick_Plotter()

# Generate spectrum with frequencies, then add same amplitudes and phases as the known 50 Hz component
ripple_freqs =  np.hstack((np.arange(10., 100., 10), np.arange(100., 600., 50), np.arange(600., 1201., 100))).ravel()
kqf_amplitudes = 9.7892e-7 * np.ones(len(ripple_freqs))
kqd_amplitudes = 9.6865e-7 * np.ones(len(ripple_freqs))
kqf_phases = 0.5564422 * np.ones(len(ripple_freqs))
kqd_phases = 0.4732764 * np.ones(len(ripple_freqs))

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

sps_kick.plot_tbt_data_to_spectrum(tbt_dict, ripple_freqs=ripple_freqs)