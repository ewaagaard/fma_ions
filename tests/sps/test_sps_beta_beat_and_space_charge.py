"""
Test if beta-beat can be added after space charge has been installed
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt

sps_fma = fma_ions.FMA(num_turns=120, n_linear=25, output_folder='Output_beta_beat_and_SC')

# Test generating a sequence with beta-beat
sps_seq = fma_ions.SPS_sequence_maker()
sps_line, twiss = sps_seq.load_xsuite_line_and_twiss(Qy_frac=19)

# Test installing space charge on this line
line_SC = sps_fma.install_SC_and_get_line(sps_line, fma_ions.BeamParameters_SPS)
twiss2 = line_SC.twiss()

# Compare difference in Twiss and beta-beat
print('\nTwiss max betx difference: {:.3f} vs {:.3f} with space charge'.format(np.max(twiss['betx']),
                                                                                np.max(twiss2['betx'])))
print('Twiss max bety difference: {:.3f} vs {:.3f} with space charge'.format(np.max(twiss['bety']),
                                                                                np.max(twiss2['bety'])))

print('X beta-beat: {:.4f}'.format( (np.max(twiss2['betx']) - np.max(twiss['betx']))/np.max(twiss['betx']) ))
print('Y beta-beat: {:.4f}\n'.format( (np.max(twiss2['bety']) - np.max(twiss['bety']))/np.max(twiss['bety']) ))


# Plot the Twiss differences
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,6))
fig.suptitle('No SC (top), SC (bottom)', fontsize=16)
ax[0].plot(twiss['s'], twiss['betx'], color='blue', label=r'$\beta_{x}$')
ax[0].plot(twiss['s'], twiss['bety'], color='orange', label=r'$\beta_{y}$')
ax[1].plot(twiss2['s'], twiss2['betx'], color='blue', label=r'$\beta_{x}$')
ax[1].plot(twiss2['s'], twiss2['bety'], color='orange', label=r'$\beta_{y}$')
ax[0].legend()
ax[1].set_xlabel('s [m]')
ax[1].set_ylabel(r'$\beta$ [m]')
ax[0].set_ylabel(r'$\beta$ [m]')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig('output_sc_test/twiss_with_sc.png', dpi=250)
plt.show()

# Test adding beta-beat to line with space charge
#line_SC_beat = sps_seq.generate_xsuite_seq_with_beta_beat(beta_beat=0.05, line=line_SC)