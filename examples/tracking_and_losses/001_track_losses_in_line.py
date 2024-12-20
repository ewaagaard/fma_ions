"""
Example script to track an SPS Pb beam in the Q26 lattice with higher intensity, and observe where particles are lost
"""
import matplotlib.pyplot as plt
import numpy as np
import fma_ions
import xtrack as xt
import xobjects as xo
import xpart as xp

# Define particle parameters
num_part = 100
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = 3.5e10 # 10 times nominal intensity to observe effect after one turn
context = xo.ContextCpu(omp_num_threads='auto')

# Load sequence - add space charge elements and beta-beat
sps_seq = fma_ions.SPS_sequence_maker(qx0=26.31, qy0=26.10)
line = sps_seq.generate_xsuite_seq(add_aperture=True)
line = sps_seq.add_beta_beat_to_line(line)
line.match(
    vary=[
        xt.Vary('kqf', step=1e-8),
        xt.Vary('kqd', step=1e-8),
        xt.Vary('qph_setvalue', step=1e-7),
        xt.Vary('qpv_setvalue', step=1e-7)
    ],
    targets = [
        xt.Target('qx', sps_seq.qx0, tol=1e-8),
        xt.Target('qy', sps_seq.qy0, tol=1e-8),
        xt.Target('dqx', sps_seq.dq1, tol=1e-7),
        xt.Target('dqy', sps_seq.dq2, tol=1e-7),
    ])
tw = line.twiss()
print('After matching: Qx = {:.4f}, Qy = {:.4f}, dQx = {:.4f}, dQy = {:.4f}\n'.format(tw['qx'], tw['qy'], tw['dqx'], tw['dqy']))
df_twiss = tw.to_pandas()

# Generate particles
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=num_part)
particles = sps.generate_particles(line, distribution_type='qgaussian', beamParams=beamParams)

# Add space charge elements
fma_sps = fma_ions.FMA()
line = fma_sps.install_SC_and_get_line(line=line,
                                        beamParams=beamParams, 
                                        optimize_for_tracking=False, 
                                        distribution_type='qgaussian', 
                                        context=context)

# Track particles
line.track(particles, num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
monitors = line.record_last_track
loss_type, loss_count = np.unique(particles.state, return_counts=True)
print('Loss types: {}, with occurrence {}'.format(loss_type, loss_count))

# Extract aperture to plot with particle trajectories
sps_plot = fma_ions.SPS_Plotting()
sv_ap, tw_ap, upperX, lowerX, upperY, lowerY, aper_idx = sps_plot.get_aperture(line, tw)

# Plot the Y particle trajectories
fig2, ax2 = plt.subplots(1, 1, figsize=(10,6), constrained_layout=True)
ax2.fill_between(tw_ap.s, upperY, lowerY, alpha=1., color='lightgrey')
ax2.plot(sv_ap.s, upperY, color="k")
ax2.plot(sv_ap.s, lowerY, color="k")
ax2.plot(monitors.s[0], monitors.x[0])