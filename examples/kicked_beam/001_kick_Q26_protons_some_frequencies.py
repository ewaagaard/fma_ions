"""
Small example to kick beam and plot TBT data for 200 macroparticles
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import PyNAFF as pnf

def get_tune_pnaf(data, turns=40):
    """
    Calculate tune using PyNAFF algorithm.
    
    Args:
        data: Turn-by-turn position data
        turns: Number of turns to analyze
    
    Returns:
        Tune value (frequency)
    """
    # Subtract the mean to remove the DC component
    data = data - np.mean(data)
    result = pnf.naff(data, turns=turns, nterms=1, skipTurns=0, getFullSpectrum=False, window=1)
    return result[0][1]  # Return the frequency (tune)


# Generate spectrum with frequencies, then add same amplitudes and phases as the known 50 Hz component
ripple_freqs = np.array([10., 50., 150., 600., 1200.])
kqf_amplitudes = 9.7892e-7 * np.ones(len(ripple_freqs))
kqd_amplitudes = 9.6865e-7 * np.ones(len(ripple_freqs))
kqf_phases = 0.5564422 * np.ones(len(ripple_freqs))
kqd_phases = 0.4732764 * np.ones(len(ripple_freqs))

# Load data and plot
sps_plot = fma_ions.SPS_Plotting()

try:
    tbt_dict = sps_plot.load_records_dict_from_json('output0_q26_protons/')
    print('Loaded dictionary\n')
    
except FileNotFoundError:
    print('Did not find dictionary, tracking!\n')
    sps = fma_ions.SPS_Flat_Bottom_Tracker(dqx0=0.0, dqy0=0.0, num_turns=20_000, num_part=200, turn_print_interval=200, proton_optics='q26',)
    tbt = sps.track_SPS(ion_type='proton', which_context='cpu', distribution_type='gaussian', install_SC_on_line=False, 
                        add_tune_ripple=True, ripple_freqs = ripple_freqs, kqf_amplitudes = kqf_amplitudes, kqd_amplitudes = kqd_amplitudes, 
                        kqf_phases=kqf_phases, kqd_phases=kqd_phases, kick_beam=True)
    tbt.to_json('output0_q26_protons/')
    tbt_dict = tbt.to_dict()

# plot turn-by-turn data
fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)
ax[0].plot(tbt_dict['X_data'], color='b')
ax[1].plot(tbt_dict['Y_data'], color='darkorange')
ax[0].set_ylabel('X [m]')
ax[1].set_ylabel('Y [m]')
ax[1].set_xlabel('Turns')


### Find FFT spectrum ###

# Analysis parameters
t4s = 40  # number of turns for NAFF analysis
i_start = 180 #200  # first turn after the kick

# Process each plane
planes = ['H', 'V']
data = {'H': tbt_dict['X_data'], 'V': tbt_dict['Y_data']}
tunes = {}

for plane in planes:
    delta = data[plane]
    i_stop = len(delta)
    
    # Analyze tunes
    tunes_singlebpm = []
    for i in range(i_start, i_stop-t4s):
        if i % 1000 == 0:
            print(f'Nr: {i}')
        # Get tune using PyNAFF
        d = delta[i:i+t4s]
        try:
            tune = get_tune_pnaf(d, turns=t4s)
        except IndexError:
            tune = np.nan
        tunes_singlebpm.append(tune)
    
    tunes[plane] = np.array(tunes_singlebpm)

# Setup plot style
integer_tune=26
colors = {'H': 'b', 'V': 'g'}
fig, (ax_tune, ax_spectrum_H, ax_spectrum_V) = plt.subplots(3, 1, figsize=(12,11), constrained_layout=True)
ax_spectrum = {
    'H': ax_spectrum_H,
    'V': ax_spectrum_V
}

for plane in planes:
    # Plot tune evolution
    turns = np.arange(len(tunes[plane]))
    ax_tune.plot(turns, integer_tune + tunes[plane], 
                label=plane, color=colors[plane])
    
    # Calculate and plot FFT of tune evolution
    N = len(tunes[plane])
    T =  2.3069302183004387e-05 # 23.03e-6 + 0.06e-6  # SPS revolution period
    Q_vals = tunes[plane][~np.isnan(tunes[plane])]
    Q_mean = np.nanmean(tunes[plane])
    yf = fftshift(fft(Q_vals - Q_mean, N))
    xf = fftshift(fftfreq(N, T))
    
    ax_spectrum[plane].semilogy(xf, 1.0/N * np.abs(yf), color=colors[plane])
    ax_spectrum[plane].set_ylabel(f'{plane} amplitude')
    ax_spectrum[plane].set_xlim(0, 1500)
    ax_spectrum[plane].set_ylim(1e-7, 1e-1)
    ax_spectrum[plane].grid(True)

    # Add markers at 50 Hz intervals
    marker_indices = [np.argmin(np.abs(xf - f)) for f in ripple_freqs]
    ax_spectrum[plane].plot(xf[marker_indices], 1.0/N * np.abs(yf)[marker_indices], 
                    'r.', markersize=8, label='50 Hz intervals')

# Finalize plots
ax_tune.set_xlabel('Turn')
ax_tune.set_ylabel('Tune')
ax_tune.legend()
ax_tune.grid(True)

ax_spectrum_V.set_xlabel('Frequency [Hz]')
plt.show()

plt.show()