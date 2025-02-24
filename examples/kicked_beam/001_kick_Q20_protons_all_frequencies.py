"""
Small example to kick beam and plot TBT data for 200 macroparticles
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt

# Generate spectrum with frequencies, then add same amplitudes and phases as the known 50 Hz component
ripple_freqs = np.hstack((np.arange(10., 100., 10), np.arange(100., 600., 50), np.arange(600., 1201., 100))).ravel()
kqf_amplitudes = 9.7892e-7 * np.ones(len(ripple_freqs))
kqd_amplitudes = 9.6865e-7 * np.ones(len(ripple_freqs))
kqf_phases = 0.5564422 * np.ones(len(ripple_freqs))
kqd_phases = 0.4732764 * np.ones(len(ripple_freqs))

sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=10_000, num_part=300, turn_print_interval=200, proton_optics='q20',)
tbt = sps.track_SPS(ion_type='proton', which_context='cpu', distribution_type='gaussian', install_SC_on_line=False, 
                    add_tune_ripple=True, ripple_freqs = ripple_freqs, kqf_amplitudes = kqf_amplitudes, add_beta_beat=True,
                    add_non_linear_magnet_errors=True, kqd_amplitudes = kqd_amplitudes, kqf_phases=kqf_phases, kqd_phases=kqd_phases, 
                    kick_beam=True, x_max_at_WS=0.025, y_max_at_WS=0.013)
tbt.to_json('output')
tbt_dict = tbt.to_dict()

# plot turn-by-turn data
fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)
ax[0].plot(tbt_dict['X_data'], color='b')
ax[1].plot(tbt_dict['Y_data'], color='darkorange')
ax[0].set_ylabel('X [m]')
ax[1].set_ylabel('Y [m]')
ax[1].set_xlabel('Turns')
plt.show()