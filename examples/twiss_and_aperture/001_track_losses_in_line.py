"""
Example script to track an SPS Pb beam in the Q26 lattice with higher intensity, and observe where particles are lost
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import fma_ions
import xtrack as xt
import xobjects as xo
import xpart as xp
import xfields as xf
import json

# Define particle parameters
num_part = 200
num_turns = 10
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = 3.5e10 # 10 times nominal intensity to observe effect after one turn
context = xo.ContextCpu(omp_num_threads='auto')

# Load sequence and save it
sps_seq = fma_ions.SPS_sequence_maker(qx0=26.31, qy0=26.10)
try:
    line = xt.Line.from_json('sps_pb_line.json')
except FileNotFoundError:
    line = sps_seq.generate_xsuite_seq(add_aperture=True)
    line.to_json('sps_pb_line.json')

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
line0 = line.copy() # make copy without space charge elements
tw0 = line0.twiss()

# Load particles if exist, otherwise generate new
try:
    # Load particles from json file to selected context
    with open('particles.json', 'r') as fid:
        particles= xp.Particles.from_dict(json.load(fid), _context=context)
except FileNotFoundError:
    sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=num_part)
    particles = sps.generate_particles(line, distribution_type='qgaussian', beamParams=beamParams)
    particles0 = particles.copy()

# Save particles
with open('particles.json', 'w') as fid:
    json.dump(particles.to_dict(), fid, cls=xo.JEncoder)

# Add space charge elements
fma_sps = fma_ions.FMA()
line = fma_sps.install_SC_and_get_line(line=line,
                                        beamParams=beamParams, 
                                        optimize_for_tracking=False, 
                                        distribution_type='qgaussian', 
                                        context=context)

# State recorder - keep track of which particles died in the last turn 
lost_particle_s = []
lost_particle_x = []
lost_particle_y = []
lost_particle_state = []

# Track particles
for i in range(num_turns):
    print('\nTurn {}'.format(i+1))
    line.track(particles, num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
    loss_type, loss_count = np.unique(particles.state, return_counts=True)

    # Save last turn, collect particles that were killed
    monitors = line.record_last_track
    monitor_counter = 0
    
    particle_state = []

    for j in range(len(monitors.state)):
        if monitors.state[j][0] == 1 and monitors.state[j][-1] < 1:
            monitor_counter += 1
            lost_particle_s.append(monitors.s[j])
            lost_particle_state.append(monitors.state[j])
            lost_particle_x.append(monitors.x[j])
            lost_particle_y.append(monitors.y[j])
    del monitors
    print('Appended {} monitors, with particles lost'.format(monitor_counter))
    print('Loss types: {}, with occurrence {}'.format(loss_type, loss_count))

# Save last tracking
monitors = line.record_last_track

# Colorcode how far killed particles got
cmap = matplotlib.colormaps['cool']
norm = plt.Normalize(vmin=monitors.s[0][0], vmax=monitors.s[0][-1])  # Set up color normalization for the colorbar, with range from 0 to bunch_range
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required to initialize the ScalarMappable, though we won't use its array

# Find where space charge elements are located
# copy line, replace collective elements with markers for stable twiss
ee0_s = []
s_coord = line.get_s_elements()
for ii, ee in enumerate(line.elements):
    if isinstance(ee, xf.SpaceChargeBiGaussian):
        ee0_s.append(s_coord[ii])
ee0_height = np.zeros(len(ee0_s))

# Extract aperture to plot with particle trajectories
sps_plot = fma_ions.SPS_Plotting()
sv_ap, tw_ap, upperX, lowerX, upperY, lowerY, aper_idx = sps_plot.get_aperture(line0, tw0)

# Plot the X and Y particle trajectories
fig, ax = plt.subplots(1, 1, figsize=(11,6), constrained_layout=True)
ax.fill_between(tw_ap.s, upperX, lowerX, alpha=1., color='lightgrey', label=None)
ax.plot(sv_ap.s, upperX, color="k", label=None)
ax.plot(sv_ap.s, lowerX, color="k", label=None)

fig2, ax2 = plt.subplots(1, 1, figsize=(11,6), constrained_layout=True)
ax2.fill_between(tw_ap.s, upperY, lowerY, alpha=1., color='lightgrey', label=None)
ax2.plot(sv_ap.s, upperY, color="k", label=None)
ax2.plot(sv_ap.s, lowerY, color="k", label=None)

# Plot alive particles
for ind in range(len(particles.y)):
    alive_ind = monitors.state[ind] > 0
    
    if np.all(alive_ind):
        ax.plot(monitors.s[ind][alive_ind], monitors.x[ind][alive_ind], color='blue', alpha=0.35, label='Alive particles' if ind==0 else None)
        ax2.plot(monitors.s[ind][alive_ind], monitors.y[ind][alive_ind], color='coral', alpha=0.35, label='Alive particles' if ind==0 else None)

# Plot particles killed in tracking
killed_at_s = []
killed_at_name = []

for i in range(len(lost_particle_s)):
    s = lost_particle_s[i]
    x = lost_particle_x[i]
    y = lost_particle_y[i]
    state = lost_particle_state[i]
    alive_ind = state > 0
    s_max =  s[alive_ind][-1]
    ele_name = df_twiss.iloc[np.abs(df_twiss['s'] - s_max).argmin()]['name']
    ax.plot(s[alive_ind], x[alive_ind], color=cmap(norm(s_max)), alpha=0.75, label='Killed particles' if i==0 else None)
    ax2.plot(s[alive_ind], y[alive_ind], color=cmap(norm(s_max)), alpha=0.75, label='Killed particles' if i==0 else None)
    killed_at_s.append(s_max)
    killed_at_name.append(ele_name )
    print('Particle {} killed at s = {:.3f} m, element name = {}'.format(i+1, s_max, ele_name))
    #print('Color: {}'.format(cmap(s_max)))

s_type, s_count = np.unique(killed_at_s, return_counts=True)
ele_type, ele_count = np.unique(killed_at_name, return_counts=True)
print('\nLoss at s type: {}, with occurrence {}'.format(s_type, s_count))
print('Loss at name type: {}, with occurrence {}'.format(ele_type, ele_count))

ax.set_ylabel('$x$ [m]')
ax2.set_ylabel('$y$ [m]')
ax.set_xlabel('$s$ [m]')
ax2.set_xlabel('$s$ [m]')

for a in [ax, ax2]:
    cbar = fig2.colorbar(sm, ax=a)
    cbar.set_label('s distance of killed particle', fontsize=15.9)   
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontsize(14.5)
    a.legend(fontsize=11, loc='upper right')
#ax2.plot(ee0_s, ee0_height, ls='None', marker='o', color='blue')
fig.savefig('SPS_Pb_X_lost_particles.png', dpi=250)
fig2.savefig('SPS_Pb_Y_lost_particles.png', dpi=250)

plt.show()