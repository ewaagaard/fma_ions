"""
Script to plot SPS Q26 beta-beat measurements with amplitude and phase method,
from Panos Zisopoulus conducted in Nov 2024
"""
import matplotlib.pyplot as plt
import numpy as np
import fma_ions
import json

# Desired RMS beta-beat values
beta_beat_x_rms_val = 0.075
beta_beat_y_rms_val = 0.07
mask_with_BPMs = True

# Load Panos beta-beat values
with open('betabeat_Q26_plot_data.json', 'rb') as handle:
    bb_dict = json.load(handle)

# Compare measured with simulated
sps_seq = fma_ions.SPS_sequence_maker(ion_type='proton', qx0=26.29, qy0= 26.26)

if mask_with_BPMs:

    simulated_label = 'Simulated - optimized RMS only at BPMs'
    
    line0 = sps_seq.generate_xsuite_seq()
    tw0 = line0.twiss()
    line = sps_seq.generate_xsuite_seq_with_beta_beat(beta_beat_x=beta_beat_x_rms_val, beta_beat_y=beta_beat_y_rms_val,
                                                      mask_with_BPMs=mask_with_BPMs)    
    tw = line.twiss()

    # Find BPM strings
    bpm_x_names = []
    bpm_y_names = []
    bpm_x_ind = []
    bpm_y_ind = []
    bpms_h = np.loadtxt('../BPMs/common_bpms_H.txt', dtype=str)
    bpms_v = np.loadtxt('../BPMs/common_bpms_V.txt', dtype=str)
    
    
    for i, key in enumerate(line.element_names):
    
        # X plane --> check BPM elements
        for bpm_h in bpms_h:
            if bpm_h[:-2].lower() in key:
                bpm_x_ind.append(i)
                bpm_x_names.append(key)
    
        # Y plane --> check BPM elements
        for bpm_v in bpms_v:
            if bpm_v[:-2].lower() in key:
                bpm_y_ind.append(i)
                bpm_y_names.append(key)
    
else:
    line0, tw0 = sps_seq.load_xsuite_line_and_twiss(save_new_xtrack_line=True)
    line = line0.copy()
    line.element_refs['qd.63510..1'].knl[1] = -1.07328640311457e-02
    line.element_refs['qf.63410..1'].knl[1] = 1.08678014669101e-02
    tw = line.twiss()

    print('After matching: Qx = {:.4f}, Qy = {:.4f}, dQx = {:.4f}, dQy = {:.4f}\n'.format(tw['qx'], tw['qy'], tw['dqx'], tw['dqy']))
    # Select all elements
    bpm_x_ind = np.arange(len(tw0))
    bpm_y_ind = np.arange(len(tw0))
    
    simulated_label = 'Simulated - optimized RMS around whole lattice'

    
# Find beta-beat in both planes, and RMS
beat_x = (tw0['betx'][bpm_x_ind] - tw['betx'][bpm_x_ind]) / tw0['betx'][bpm_x_ind] 
beat_y = (tw0['bety'][bpm_y_ind] - tw['bety'][bpm_y_ind]) / tw0['bety'][bpm_y_ind] 

rms_x = np.sqrt(np.sum(beat_x**2)/len(beat_x))
rms_y = np.sqrt(np.sum(beat_y**2)/len(beat_y))
print('\nActual beta-beat values:')
print('\nRMS value X: {:.5f}'.format(rms_x))
print('RMS value Y: {:.5f}'.format(rms_y))
print('Peak value X: {:.5f}'.format(np.max(beat_x)))
print('Peak value Y: {:.5f}'.format(np.max(beat_y)))

# Generate figure
fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True, constrained_layout=True)
ax[0].errorbar(bb_dict['sx_amplitude'], bb_dict['dbb_H_amplitude_avg'], yerr=bb_dict['dbb_H_amplitude_std'], fmt='o', label='Amplitude method')
ax[0].errorbar(bb_dict['sx_phase'], bb_dict['dbb_H_phase_avg'], yerr=bb_dict['dbb_H_phase_std'], fmt='o', label='Phase method')
ax[1].errorbar(bb_dict['sx_amplitude'], bb_dict['dbb_H_amplitude_avg'], yerr=bb_dict['dbb_H_amplitude_std'], fmt='o', label='Amplitude method')
ax[1].errorbar(bb_dict['sx_phase'], bb_dict['dbb_H_phase_avg'], yerr=bb_dict['dbb_H_phase_std'], fmt='o', label='Phase method')
ax[0].set_ylabel('$\Delta \\beta_{x} / \\beta_{x}$ [%]')
ax[1].set_ylabel('$\Delta \\beta_{y} / \\beta_{y}$ [%]')
for a in ax:
    a.legend(fontsize=10.5)
    a.grid(alpha=0.55)
ax[1].set_xlabel('s [m]')
fig.savefig('plots/SPS_Q26_ion_tunes_beta_beat_measurements.png', dpi=250)

# Generate figure
amp_color = 'cyan'
fig1, ax1 = plt.subplots(2,1, figsize=(10,6), sharex=True, constrained_layout=True)
ax1[0].errorbar(bb_dict['sx_amplitude'], bb_dict['dbb_H_amplitude_avg'], yerr=bb_dict['dbb_H_amplitude_std'], marker='o', ls='None', color='k', ecolor=amp_color, markerfacecolor=amp_color, label='Amplitude method')
ax1[0].errorbar(bb_dict['sx_phase'], bb_dict['dbb_H_phase_avg'], yerr=bb_dict['dbb_H_phase_std'], marker='s', color='k', ls='None', ecolor='darkorange', markerfacecolor='darkorange', label='Phase method')
ax1[1].errorbar(bb_dict['sx_amplitude'], bb_dict['dbb_H_amplitude_avg'], yerr=bb_dict['dbb_H_amplitude_std'], marker='o', color='k', ls='None', ecolor=amp_color,  markerfacecolor=amp_color, label='Amplitude method')
ax1[1].errorbar(bb_dict['sx_phase'], bb_dict['dbb_H_phase_avg'], yerr=bb_dict['dbb_H_phase_std'], marker='s', color='k', ls='None', ecolor='darkorange', markerfacecolor='darkorange', label='Phase method')
ax1[0].set_ylabel('$\Delta \\beta_{x} / \\beta_{x}$ [%]')
ax1[1].set_ylabel('$\Delta \\beta_{y} / \\beta_{y}$ [%]')

if mask_with_BPMs:
    ax1[0].plot(tw0.s[bpm_x_ind], 100*beat_x, color='green', marker='v', label=simulated_label)
    ax1[1].plot(tw0.s[bpm_y_ind], 100*beat_y, color='green', marker='v', label=simulated_label)
else:
    ax1[0].plot(tw0.s[bpm_x_ind], 100*beat_x, color='green', marker='v', label=simulated_label)
    ax1[1].plot(tw0.s[bpm_y_ind], 100*beat_y, color='green', marker='v', label=simulated_label)

for a in ax1:
    a.legend(fontsize=11.5)
    a.grid(alpha=0.55)
ax1[1].set_xlabel('s [m]')
    
if mask_with_BPMs:
    fig1.savefig('plots/SPS_Q26_ion_tunes_beta_beat_simulated_all_lattice_measurements.png', dpi=250)
else:
    fig1.savefig('plots/SPS_Q26_ion_tunes_beta_beat_simulated_only_at_BPMs_measurements.png', dpi=250)

plt.show()
