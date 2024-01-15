"""
Generate SPS Pb ion FMA plot - symmetric lattice without QFA and QDA, no momentum offset
and then 
- Generate particle distribution with fixed Jy and varying Jx - "trace"" rather than "grid""
- Track particles and find tunes and corresponding action (and normalized beam size)
- Plot tune over action, and normalized phase space 
- Gradually increase QD error to observe if 8 islands in phase space become 4 (8Qx becoming 4Qx)
"""
import fma_ions
import numpy as np

# Minimum diffusion threshold
d_min = -11.5

# Beta-beat array: 0.1% to 5%
beats = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])

# Load Twiss and plot normalized phase space
sps_seq = fma_ions.SPS_sequence_maker()
sps_line, twiss_sps = sps_seq.load_xsuite_line_and_twiss(Qy_frac=25, use_symmetric_lattice=True)

# Test case without QD error
fma_sps = fma_ions.FMA(output_folder='output_Pb_symmetric_lattice_QD_0', z0=0., n_linear=60, r_min=2.9, n_sigma=3.2)
fma_sps.run_SPS(load_tbt_data=True, make_single_Jy_trace=True, use_symmetric_lattice=True)

Jx, Jy, Qx, Qy, d = fma_sps.plot_tune_over_action(twiss_sps, load_tune_data=True, also_show_plot=False, case_name='z0 = 0')

# Select index with a diffusion coefficient higher than a certain threshold
ind = np.where(d > d_min)[0]

fma_sps.plot_normalized_phase_space(twiss_sps, particle_index=ind, also_show_plot=False, case_name='beat = 0.0')


for i, beat in enumerate(beats):
    print('\nStarting beat {}!\n'.format(beat))
    
    fma_sps_beats = fma_ions.FMA(output_folder='output_Pb_symmetric_lattice_QD_{}'.format(i+1), z0=0., n_linear=60, r_min=2.9, n_sigma=3.2)
    fma_sps_beats.run_SPS_with_beta_beat(load_tbt_data=True, beta_beat=beat,
                                         make_single_Jy_trace=True, use_symmetric_lattice=True)

    Jx, Jy, Qx, Qy, d = fma_sps_beats.plot_tune_over_action(twiss_sps, load_tune_data=True, also_show_plot=False, case_name='Y_beat_{}'.format(beat))

    # Select index with a diffusion coefficient higher than a certain threshold
    ind = np.where(d > d_min)[0]

    fma_sps_beats.plot_normalized_phase_space(twiss_sps, particle_index=ind, also_show_plot=False, case_name='Y_beat_{}'.format(beat))
