"""
Test script to generate tune ripple for several frequencies: 50, 150 and 300 Hz
"""
import numpy as np
import fma_ions
import matplotlib.pyplot as plt
import scipy

# Provide parameters 
turns_per_sec = 42969
ripple_periods = np.array([859, 286, 143]) # 50, 150 and 300 Hz
freqs = [50, 150, 300]
kqf_amplitudes = np.array([9.7892e-7, 2.2421e-7, 3.1801e-7])
kqd_amplitudes = np.array([9.6865e-7, 4.4711e-7, 5.5065e-7])
kqf_phases = np.array([0.5564486, 1.3804139, -3.0285897]) * turns_per_sec
kqd_phases = np.array([0.47329223, -2.018479, -3.1261365]) * turns_per_sec

# Generate tune ripple object
turns = np.arange(10_000)
ripple_maker = fma_ions.Tune_Ripple_SPS(num_turns=10_000)
kqf_ripple, kqd_ripple = ripple_maker.get_k_ripple_summed_signal(ripple_periods, kqf_amplitudes, kqd_amplitudes,
                                                                 kqd_phases=kqd_phases, kqf_phases=kqf_phases)

# Make FFT on ripple to find correct frequencies
N = len(kqf_ripple)
fft_kqd = scipy.fft.fft(kqd_ripple)
fft_kqf = scipy.fft.fft(kqf_ripple)
FFT_QD = scipy.fft.fftshift(abs(fft_kqd)) / N # normalized magnitude
FFT_QF = scipy.fft.fftshift(abs(fft_kqf)) / N # normalized magnitude
freqs = scipy.fft.fftshift(scipy.fftpack.fftfreq(kqd_ripple.size)) # one point, one turn

# Plot the signal
# Plot raw current
fig0, ax = plt.subplots(2,1, figsize=(8, 7.5), sharex=True, constrained_layout=True)
ax[0].plot(turns, kqd_ripple, label='kqd - reconstructed ripple 50, 150 and 300 Hz', alpha=0.65)
ax[1].plot(turns, kqf_ripple, label='kqf - reconstructed ripple 50, 150 and 300 Hz', alpha=0.65)
plt.grid(alpha=0.55)
for a in ax:
    a.legend(fontsize=10)
ax[0].set_ylabel('kqd [m$^{-2}$]')
ax[1].set_ylabel('kqf [m$^{-2}$]')
ax[1].set_xlabel('Turns')

fig1, ax1 = plt.subplots(2,1, figsize=(8,6), sharex=True, constrained_layout=True)
ax1[0].plot(freqs, FFT_QD, label='$k_{\mathrm{QD}}$ - reconstructed ripple')
ax1[1].plot(freqs, FFT_QF, label='$k_{\mathrm{QD}}$ - reconstructed ripple')
for period in ripple_periods:
    for a in ax1:
        a.axvline(1/period, color='red')
#ax.set_ylim(1e-4, 0.2)
for a in ax1:
    a.set_xlim(-0.0001, 0.01)
    a.set_ylabel('Norm FFT mag.', fontsize=16)
ax1[1].set_xlabel('Noise frequency [1/turn]', fontsize=16)

plt.show()