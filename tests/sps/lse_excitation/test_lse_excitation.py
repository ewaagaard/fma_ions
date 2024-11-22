"""
Small script to test the LSE excitation in the SPS
"""
import fma_ions
import matplotlib.pyplot as plt

# Instantiate flat bottom tracker object - set high horizontal tune
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0=26.36, qy0=26.19, num_part=200, num_turns=200, turn_print_interval=10)
tbt = sps.track_SPS(install_SC_on_line=True, beta_beat=0.15, add_non_linear_magnet_errors=True, 
                    I_LSE=-10.)
print('Finished LSE excitation test')

sps_plot = fma_ions.SPS_Plotting()
tbt_dict = tbt.to_dict(convert_to_numpy=True)

# Emittances and bunch intensity 
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (9.5, 3.6), constrained_layout=True)
ax1.plot(tbt_dict['Seconds'], tbt_dict['exn'] * 1e6, alpha=0.7, c='turquoise', lw=1.5, label='Simulated')
ax2.plot(tbt_dict['Seconds'], tbt_dict['eyn'] * 1e6, alpha=0.7, c='turquoise', lw=1.5, label='Simulated')
ax3.plot(tbt_dict['Seconds'], tbt_dict['Nb'], alpha=0.7, lw=2.2, c='turquoise', label='Simulated')
for a in [ax1, ax2, ax3]:
    a.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\varepsilon_{x}^{n}$ [$\mu$m]')
ax2.set_ylabel(r'$\varepsilon_{y}^{n}$ [$\mu$m]')
ax3.set_ylabel(r'Ions per bunch $N_{b}$')
plt.show()