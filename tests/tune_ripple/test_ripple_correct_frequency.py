"""
Tester script to ensure that a specified frequency of 1200 Hz corresponds to this frequency
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import fma_ions

# Specify parameters for turns and ripple frequency
num_turns = 100_000
ripple_freqs = np.array([1200.])
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
ind = np.where(xf > 40.)

# Plot the signal and FFT spectrum - turns
fig, ax = plt.subplots(2, 1, figsize=(8,6), constrained_layout=True)
ax[0].plot(time, kqf_ripple, label='kqf')
ax[0].plot(time, kqd_ripple, alpha=0.8, label='kqd')
ax[0].set_xlabel('Turns')
ax[0].set_ylabel('k amplitude')
ax[0].legend()
ax[1].semilogy(xf[ind], yf_kqf[ind], marker='o', label='kqf')
ax[1].semilogy(xf[ind], yf_kqd[ind], marker='o', alpha=0.8, label='kqd')
for a in ax:
    a.grid(alpha=0.55)
ax[1].set_xlim(1000., 1400.)
ax[1].set_xlabel('Freqeuncy [Hz]')
ax[1].set_ylabel('Norm FFT amplitude')
plt.show()