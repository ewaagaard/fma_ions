"""
Script to generate SPS sequence with correct beta-beat
Panos measured in Nov 2024 for Q26 protons, which we emulate here

Then plot beta-beat distribution around the ring
"""
import matplotlib.pyplot as plt
import numpy as np
import fma_ions
import json

# Load Panos beta-beat values
with open('betabeat_Q26_plot_data.json', 'rb') as handle:
    bb_dict = json.load(handle)


# Desired RMS beta-beat values
beta_beat_x_rms_val = 0.075
beta_beat_y_rms_val = 0.07

# Create SPS sequence maker
sps_seq = fma_ions.SPS_sequence_maker(ion_type='proton', qx0=26.29, qy0= 26.26)

# Generate sequences with and without beta-beat
line0 = sps_seq.generate_xsuite_seq()
tw0 = line0.twiss()
line = sps_seq.generate_xsuite_seq_with_beta_beat(beta_beat_x=beta_beat_x_rms_val, beta_beat_y=beta_beat_y_rms_val,
                                                  mask_with_BPMs=False)
tw = line.twiss()

print('After matching: Qx = {:.4f}, Qy = {:.4f}, dQx = {:.4f}, dQy = {:.4f}\n'.format(tw['qx'], tw['qy'], tw['dqx'], tw['dqy']))
print('Achieved with new single quadrupolar knobs:')
print('kk_QD = {:.14e}, kk_QF = {:.14e}'.format(line.vars['kk_QD']._value, line.vars['kk_QF']._value))

# Find beta-beat in both planes, and RMS
beat_x = (tw0['betx'][sps_seq.bpm_x_ind] - tw['betx'][sps_seq.bpm_x_ind]) / tw0['betx'][sps_seq.bpm_x_ind] 
beat_y = (tw0['bety'][sps_seq.bpm_y_ind] - tw['bety'][sps_seq.bpm_y_ind]) / tw0['bety'][sps_seq.bpm_y_ind] 

rms_x = np.sqrt(np.sum(beat_x**2)/len(beat_x))
rms_y = np.sqrt(np.sum(beat_y**2)/len(beat_y))
print('\nActual beta-beat values:')
print('\nRMS value X: {:.5f}'.format(rms_x))
print('RMS value Y: {:.5f}'.format(rms_y))
print('Peak value X: {:.5f}'.format(np.max(beat_x)))
print('Peak value Y: {:.5f}'.format(np.max(beat_y)))

# Find original beta functions and k-values of quadrupolar slice with knobs
aa = tw0.to_pandas()
qd0 = aa[aa['name'] == 'qd.63510..1']
betx_qd0 = qd0.betx.values[0]
bety_qd0 = qd0.bety.values[0]
qf0 = aa[aa['name'] == 'qf.63410..1']
betx_qf0 = qf0.betx.values[0]
bety_qf0 = qf0.bety.values[0]

dk_qd = np.abs(line.element_refs['qd.63510..1'].knl[1]._value - line0.element_refs['qd.63510..1'].knl[1]._value)
dk_qf = np.abs(line.element_refs['qf.63410..1'].knl[1]._value - line0.element_refs['qf.63410..1'].knl[1]._value)

# Compare with theoretical formula for RMS beta-beat from single quadrupolar error - plug in RMS beta-beat from above
rms_x_theoretical = 1 / (2*np.sqrt(2)*np.abs(np.sin(2*np.pi*tw0['qx']))) * np.sqrt( (dk_qd*betx_qd0 )**2  + (dk_qf*betx_qf0 )**2)
rms_y_theoretical = 1 / (2*np.sqrt(2)*np.abs(np.sin(2*np.pi*tw0['qy']))) * np.sqrt( (dk_qd*bety_qd0 )**2  + (dk_qf*bety_qf0 )**2)
print('\nTheoretical RMS beta-veat value from dk at qd.63510..1 and qf.63410..1')
print('RMS value X: {:.5f}'.format(rms_x_theoretical))
print('RMS value XY {:.5f}'.format(rms_y_theoretical))

# Plot the beta-beat
fig, ax = plt.subplots(2,1, figsize=(8,6), sharex=True, constrained_layout=True)
ax[0].plot(tw0.s[sps_seq.bpm_x_ind], 100*beat_x, color='blue')
ax[1].plot(tw0.s[sps_seq.bpm_y_ind], 100*beat_y, color='darkorange')
ax[0].set_ylabel('$\Delta \\beta_{x} / \\beta_{x}$ [%]')
ax[1].set_ylabel('$\Delta \\beta_{y} / \\beta_{y}$ [%]')
ax[1].set_xlabel('s [m]')
for a in ax:
    a.grid(alpha=0.55)
plt.show()