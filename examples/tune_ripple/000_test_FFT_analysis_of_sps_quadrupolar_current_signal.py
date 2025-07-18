"""
Script to check FFT of SPS quadrupolar circuit currents
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Simulation parameters
fs = 1000  # Sampling frequency (Hz)
duration = 2  # Duration in seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
N = len(t)

# Signal parameters
I_dc = 70.0  # DC current (A)
I_50hz = 0.001  # 50 Hz component amplitude (A) - at the threshold
I_100hz = 0.0 #0.0005  # 100 Hz harmonic (A)
I_150hz = 0.0#0.0003  # 150 Hz harmonic (A)

# Add some broadband noise to simulate other fluctuations
noise_level = 0.0#0.003  # A (RMS)

# Create synthetic DCCT signal
current = (I_dc + 
           I_50hz * np.sin(2 * np.pi * 50 * t) + 
           I_100hz * np.sin(2 * np.pi * 100 * t + 0.5) + 
           I_150hz * np.sin(2 * np.pi * 150 * t + 1.2) + 
           noise_level * np.random.randn(N))

print(f"Signal statistics:")
print(f"Mean current: {np.mean(current):.4f} A")
print(f"Current fluctuations (std): {np.std(current):.4f} A")
print(f"Peak-to-peak variation: {np.ptp(current):.4f} A")

# Remove DC component (subtract mean)
current_ac = current - np.mean(current)

# Perform FFT
fft_result = fft(current_ac)
frequencies = fftfreq(N, 1/fs)

# Calculate normalized FFT amplitude (amplitude/N)
fft_amplitude_normalized = np.abs(fft_result) / N

# Only consider positive frequencies
pos_freq_mask = frequencies >= 0
frequencies_pos = frequencies[pos_freq_mask]
fft_amplitude_pos = fft_amplitude_normalized[pos_freq_mask]

# Find the 50 Hz component
freq_50hz_idx = np.argmin(np.abs(frequencies_pos - 50))
amplitude_50hz_normalized = fft_amplitude_pos[freq_50hz_idx]
amplitude_50hz_actual = amplitude_50hz_normalized * 2  # Convert back to actual amplitude

print(f"\nFFT Analysis:")
print(f"50 Hz normalized FFT amplitude: {amplitude_50hz_normalized:.6f}")
print(f"50 Hz actual current amplitude: {amplitude_50hz_actual:.6f} A")
print(f"Threshold (5e-4): {'PASS' if amplitude_50hz_normalized < 5e-4 else 'FAIL'}")

# Create plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Time domain plot
ax1.plot(t[:500], current[:500], 'b-', linewidth=0.8)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Current (A)')
ax1.set_title('DCCT Current Measurement (first 0.5 seconds)')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([I_dc - 0.02, I_dc + 0.02])

# AC component time domain
ax2.plot(t[:500], current_ac[:500], 'r-', linewidth=0.8)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('AC Current (A)')
ax2.set_title('AC Component (DC removed)')
ax2.grid(True, alpha=0.3)

# Frequency domain plot
ax3.semilogy(frequencies_pos[:N//10], fft_amplitude_pos[:N//10], 'g-', linewidth=1)
ax3.axhline(y=5e-4, color='r', linestyle='--', label='Threshold (5e-4)')
ax3.axvline(x=50, color='orange', linestyle=':', label='50 Hz')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Normalized FFT Amplitude')
ax3.set_title('FFT Analysis (Normalized to Number of Samples)')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_xlim([0, 200])
ax3.set_ylim([1e-6, 1e-2])

plt.tight_layout()
plt.show()

# Additional analysis: Find all significant peaks
significant_peaks = []
for i, freq in enumerate(frequencies_pos):
    if 1 <= freq <= 500 and fft_amplitude_pos[i] > 1e-5:  # Above noise floor
        significant_peaks.append((freq, fft_amplitude_pos[i], fft_amplitude_pos[i] * 2))

print(f"\nSignificant frequency components:")
print(f"{'Frequency (Hz)':<15} {'Normalized Amp':<15} {'Actual Amp (A)':<15}")
print("-" * 45)
for freq, norm_amp, actual_amp in sorted(significant_peaks, key=lambda x: x[1], reverse=True)[:10]:
    print(f"{freq:<15.1f} {norm_amp:<15.6f} {actual_amp:<15.6f}")

# Compensation quality assessment
print(f"\nCompensation Quality Assessment:")
print(f"50 Hz component: {amplitude_50hz_actual*1000:.3f} mA")
if amplitude_50hz_normalized < 5e-4:
    print("✓ Good compensation achieved (below threshold)")
else:
    print("✗ Poor compensation (above threshold)")
    
print(f"\nFor reference:")
print(f"- Total RMS fluctuation: {np.std(current_ac)*1000:.2f} mA")
print(f"- 50 Hz component represents {(amplitude_50hz_actual/np.std(current_ac)*100):.1f}% of total fluctuation")
plt.show()
