"""
Small example to kick beam and plot TBT data for 200 macroparticles
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt

output_folder = 'output_Q26_Pb_10k'

# Generate spectrum with frequencies, then add same amplitudes and phases as the known 50 Hz component
ripple_freqs = np.hstack((np.arange(10., 100., 10), np.arange(100., 600., 50), np.arange(600., 1201., 100))).ravel()
kqf_amplitudes = 9.7892e-7 * np.ones(len(ripple_freqs))
kqd_amplitudes = 9.6865e-7 * np.ones(len(ripple_freqs))
kqf_phases = 0.5564422 * np.ones(len(ripple_freqs))
kqd_phases = 0.4732764 * np.ones(len(ripple_freqs))

# Load data and plot
sps_plot = fma_ions.SPS_Plotting()
sps_kick = fma_ions.SPS_Kick_Plotter()

try:
    tbt_dict = sps_plot.load_records_dict_from_json(output_folder)
    print('Loaded dictionary\n')
    
except FileNotFoundError:
    print('Did not find dictionary, tracking!\n')
    sps = fma_ions.SPS_Flat_Bottom_Tracker(dqx0=0.0, dqy0=0.0, num_turns=10_000, turn_print_interval=500)
    tbt = sps.track_SPS(which_context='cpu', distribution_type='single', install_SC_on_line=False, 
                        add_tune_ripple=True, ripple_freqs = ripple_freqs, kqf_amplitudes = kqf_amplitudes,
                        kqd_amplitudes = kqd_amplitudes, kqf_phases=kqf_phases, kqd_phases=kqd_phases, kick_beam=True)
    tbt.to_json(output_folder)
    tbt_dict = tbt.to_dict()

# plot turn-by-turn data
fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)
ax[0].plot(tbt_dict['X_data'], color='b')
ax[1].plot(tbt_dict['Y_data'], color='darkorange')
ax[0].set_ylabel('X [m]')
ax[1].set_ylabel('Y [m]')
ax[1].set_xlabel('Turns')

sps_kick.plot_tbt_data_to_spectrum(output_folder=output_folder, ripple_freqs=ripple_freqs)