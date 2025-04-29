"""
Tester script to ensure that a specified frequencies between 10 and 1200 Hz corresponds to correct
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import fma_ions

### STEP 1: check spectrum of kqf/kqd signal manually ###

# Specify parameters for turns and ripple frequency
num_turns = 100_000
ripple_freqs = np.hstack((np.arange(10., 100., 10), np.arange(100., 600., 50), np.arange(600., 1201., 100))).ravel()
kqf_amplitudes = 9.7892e-7 * np.ones(len(ripple_freqs))
kqd_amplitudes = 9.6865e-7 * np.ones(len(ripple_freqs))
kqf_phases = 0.5564422 * np.ones(len(ripple_freqs))
kqd_phases = 0.4732764 * np.ones(len(ripple_freqs))

# Create ripple in quadrupolar knobs, convert phases to turns
turns_per_sec = 42968.72568137877 # from Q26 Pb revolution frequency
ripple_periods = turns_per_sec/ripple_freqs #).astype(int)  # number of turns particle makes during one ripple oscillation
kqf_phases_turns = kqf_phases * turns_per_sec # convert time domain to turn domain, i.e. multiply with turns/sec
kqd_phases_turns = kqd_phases * turns_per_sec # convert time domain to turn domain, i.e. multiply with turns/sec

ripple_maker = fma_ions.Tune_Ripple_SPS(num_turns=num_turns)
kqf_ripple, kqd_ripple = ripple_maker.get_k_ripple_summed_signal(ripple_periods, kqf_amplitudes, kqd_amplitudes,
                                                                 kqf_phases_turns, kqd_phases_turns)
N = len(kqf_ripple)
turns = np.arange(N)
time = turns / turns_per_sec

# Compute spectrum in time
yf_kqf = np.abs(fftshift(fft(kqf_ripple - np.nanmean(kqf_ripple), N))) / N
yf_kqd = np.abs(fftshift(fft(kqd_ripple - np.nanmean(kqd_ripple), N))) / N
xf = fftshift(fftfreq(N, time[1] - time[0]))
ind = np.where(xf > 0.)

# Plot the signal and FFT spectrum - turns
fig, ax = plt.subplots(2, 1, figsize=(12,6), constrained_layout=True)
ax[0].plot(time, kqf_ripple, label='kqf')
ax[0].plot(time, kqd_ripple, alpha=0.8, label='kqd')
ax[0].set_xlabel('Turns')
ax[0].set_ylabel('k amplitude')
ax[0].legend()
ax[1].semilogy(xf[ind], yf_kqf[ind], marker='o', ms=2, label='kqf')
ax[1].semilogy(xf[ind], yf_kqd[ind], marker='o', ms=2, alpha=0.8, label='kqd')

# Add markers at 50 Hz intervals
marker_indices = [np.argmin(np.abs(xf - f)) for f in ripple_freqs]
ax[1].plot(xf[marker_indices], np.abs(yf_kqf)[marker_indices], 
                'r.', markersize=8, label='50 Hz intervals')

for a in ax:
    a.grid(alpha=0.55)
ax[1].set_xlim(0., 1250.)
ax[1].set_xlabel('Freqeuncy [Hz]')
ax[1].set_ylabel('Norm FFT amplitude')
plt.show()

### STEP 2: check FFT spectrum of TUNES from kqf/kqd spectrum ###
sps_kick = fma_ions.SPS_Kick_Plotter()
Qx_knobs, Qy_knobs = sps_kick.knobs_to_tunes(kqd_ripple, kqf_ripple)

yf_qx = np.abs(fftshift(fft(Qx_knobs - np.nanmean(Qx_knobs), N))) / N
yf_qy = np.abs(fftshift(fft(Qy_knobs - np.nanmean(Qy_knobs), N))) / N

fig2, ax2 = plt.subplots(2, 1, figsize=(12,6), constrained_layout=True)
ax2[0].plot(time, Qx_knobs, label='Qx')
ax2[0].plot(time, Qy_knobs, alpha=0.8, label='Qy')
ax2[0].set_xlabel('Turns')
ax2[0].set_ylabel('Tunes')
ax2[0].legend()
ax2[1].semilogy(xf[ind], yf_qx[ind], marker='o', ms=2, label='qx')
ax2[1].semilogy(xf[ind], yf_qy[ind], marker='o', ms=2, alpha=0.8, label='qy')
ax2[1].plot(xf[marker_indices], np.abs(yf_qx)[marker_indices], 
                'r.', markersize=8, label='50 Hz intervals')

for a in ax2:
    a.grid(alpha=0.55)
ax2[1].set_xlim(0., 1250.)
ax2[1].set_xlabel('Freqeuncy [Hz]')
ax2[1].set_ylabel('Norm FFT amplitude')
plt.show()

'''
### STEP 3: Test with PyNAFF ###
tunes_knob = {'Qx_knobs': Qx_knobs, 'Qy_knobs': Qy_knobs}

# Setup plot style
colors = {'H': 'b', 'V': 'g'}
colors2 = ['cyan', 'lime']
fig, (ax_tune, ax_spectrum_H, ax_spectrum_V) = plt.subplots(3, 1, figsize=(12,11), constrained_layout=True)
ax_spectrum = {
    'H': ax_spectrum_H,
    'V': ax_spectrum_V
}
planes = ['H', 'V']
t4s = 40
i_start = 200
T = 2.327274044418234e-05 # Pb ions

# FFT from current spectrum 
for i, key in enumerate(tunes_knob):
    ax_tune.plot(turns+t4s/2, tunes_knob[key], 
                label=f'{planes[i]} tune from knobs k', alpha=0.85, color=colors2[i])
    N_knob = len(tunes_knob[key][ind])
    yf_knob = np.abs(fftshift(fft(tunes_knob[key] - np.nanmean(tunes_knob[key]), N_knob))) / N_knob
    xf_knob = fftshift(fftfreq(N_knob, T))
    ind = np.where(xf_knob > 0.)
    
    ax_spectrum[planes[i]].semilogy(xf_knob[ind], yf_knob[ind], ls='--', alpha=0.85, color=colors2[i], label='Knobs k tune spectrum')
    ax_spectrum[planes[i]].legend(fontsize=13)
    ax_spectrum[planes[i]].grid(alpha=0.55)
    # Add markers at 50 Hz intervals
    marker_indices = [np.argmin(np.abs(xf_knob - f)) for f in ripple_freqs]
    ax_spectrum[planes[i]].plot(xf_knob[marker_indices], np.abs(yf_knob)[marker_indices], 
                        'r.', markersize=8, label='50 Hz intervals')
    ax_spectrum[planes[i]].set_xlim(0, 1500)
    #ax_spectrum['H'].set_ylim(1e-5, 1e-2)
ax_tune.set_title('Tune evolution TBT vs knobs data')
ax_tune.set_xlabel('Turn')
ax_tune.set_ylabel('Tune')
plt.show()
'''