"""
Small example to calculate the effect of beam pipe shielding in the SPS
inspired by Section 4.5 in Frank Zimmermann's thesis (1993)
"""
import numpy as np
import matplotlib.pyplot as plt
import fma_ions

def calculate_beam_pipe_shielding(f, r, d, sigma, mu0=4*np.pi*1e-7):
    """
    Calculates the magnitude of the magnetic field shielding factor of a beam pipe.

    Args:
        f (np.ndarray or float): Frequency in Hz.
        r (float): Beam pipe inner radius in meters.
        d (float): Beam pipe wall thickness in meters.
        sigma (float): Beam pipe material conductivity in S/m.
        mu0 (float): Magnetic permeability of free space in T m/A.

    Returns:
        np.ndarray or float: Magnitude of the shielding factor |Q_tilde(f)|.
                              Returns 1 for f=0 to avoid division by zero in skin depth.
    """
    # Ensure f is not zero to avoid division by zero in skin depth calculation
    f = np.maximum(f, 1e-9)

    # Calculate skin depth (Eq. 4.15)
    delta = 1.0 / np.sqrt(np.pi * f * mu0 * sigma)

    # Calculate intermediate complex variable k(f)
    k = (1 + 1j) / delta

    # Calculate K(f)
    K = k * (r - d)

    # Calculate the argument of cosh and sinh
    arg = k * d

    # Calculate cosh and sinh terms
    cosh_term = np.cosh(arg)
    sinh_term = np.sinh(arg)

    # Calculate the [K + 1/K] term
    K_plus_invK = K + 1.0 / K

    # Calculate the denominator term
    denominator = cosh_term + 0.5 * K_plus_invK * sinh_term

    # Calculate the shielding factor Q_tilde(f)
    Q_tilde = 1.0 / denominator

    # Return the magnitude
    return np.abs(Q_tilde)

# --- Example Usage ---
# SPS warm section parameters
sps_r = 0.04 # at SPS quads horizontally #0.02  # 2 cm
sps_d = 0.002 # 2 mm
sps_sigma = 3e6 # 3e6 S/m for stainless steel

# Frequency range for illustration
frequencies_shielding = np.logspace(0, 4, 500) # From 1 Hz to 1000 Hz

# Calculate shielding factor magnitude
shielding_factor_magnitude = calculate_beam_pipe_shielding(
    frequencies_shielding,
    sps_r,
    sps_d,
    sps_sigma
)

# Plot the shielding factor (log-log scale)
f, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
ax.plot(frequencies_shielding, shielding_factor_magnitude)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('|$\\tilde{Q}(f)|$') # Shielding Factor Magnitude 
#plt.title('Illustrative Beam Pipe Shielding Factor (SPS Parameters)')
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_ylim(0., 1.)
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.show()