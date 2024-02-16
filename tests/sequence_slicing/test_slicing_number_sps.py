"""
Compare tracking of 1 vs 5 slices of sequence
"""
import fma_ions
import matplotlib.pyplot as plt
import numpy as np
import time
import xpart as xp
import xtrack as xt
import xobjects as xo

context = xo.ContextCpu()

n_part = 100
n_turns = 100

# Generate sequence with 1 slice and 5 slices
sps = fma_ions.SPS_sequence_maker()
line1 = sps.generate_xsuite_seq(nr_slices=1)
line2 = sps.generate_xsuite_seq(nr_slices=5)

### Compare non-linear chromatic behaviour of ring 
delta_values = np.arange(-0.006, 0.006, 0.001)
qx_values = np.zeros([2, len(delta_values)])
qy_values = np.zeros([2, len(delta_values)])

#### FIRST TEST - TRACKING #### 
particles0 = xp.generate_matched_gaussian_bunch(_context=context,
        num_particles=n_part, total_intensity_particles=fma_ions.BeamParameters_SPS.Nb,
        nemitt_x=fma_ions.BeamParameters_SPS.exn, nemitt_y=fma_ions.BeamParameters_SPS.eyn, 
        sigma_z= fma_ions.BeamParameters_SPS.sigma_z,
        particle_ref=line1.particle_ref, line=line1)

particles1 = xp.Particles(mass0=particles0.mass0, gamma0=particles0.gamma0[0], q0 = 82., 
                           x=[1e-6], px=[1e-6], y=[1e-6], py=[1e-6],
                           zeta=[0], delta=[0])
particles2 = particles1.copy()

# Do the tracking and clock
time00 = time.time()
for turn in range(n_turns):
   line1.track(particles1, num_turns=n_turns)
   if turn % 5 == 0:
       print('1 Tracking turn {}'.format(turn))
time01 = time.time()
dt0 = time01-time00

time10 = time.time()
for turn in range(n_turns):
   line2.track(particles2)
   if turn % 5 == 0:
       print('2 Tracking turn {}'.format(turn))
time11 = time.time()
dt1 = time11-time10

print(np.abs(particles1.x - particles2.x))    
print('\n1 slice tracking: {} s\n5 slices tracking: {} s'.format(dt0, dt1))
print('\nDifference in final x coord:')
print(np.abs(particles1.x - particles2.x))

#### SECOND TEST - NON-LINEAR CHROMATIC CURVES ####
for i, delta in enumerate(delta_values):
    print(f"\nXtrack Working on {i} of delta values {len(delta_values)}")
    # Xtrack
    print("Testing Xtrack twiss...")
    tt = line1.twiss(method='4d', delta0=delta)
    qx_values[0, i] = tt.qx
    qy_values[0, i] = tt.qy
    
    tt_2 = line2.twiss(method='4d', delta0=delta)
    qx_values[1, i] = tt_2.qx
    qy_values[1, i] = tt_2.qy
    
# Plot the result
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,7))
fig.suptitle('SPS Pb ions: Non-linear chromatic tests')
ax1.plot(delta_values, qx_values[0, :], marker='o', c='r', label='Xtrack - 1 slice')
ax1.plot(delta_values, qx_values[1, :], marker='v', c='darkred', alpha=0.7, label='Xtrack - 5 slices')
ax1.legend(fontsize=14)
ax1.set_xticklabels([])
ax1.set_ylabel('$Q_{x}$')
ax2.plot(delta_values, qy_values[0, :], marker='o', c='r', label='Xtrack - 1 slice')
ax2.plot(delta_values, qy_values[1, :], marker='v', c='darkred', alpha=0.7, label='Xtrack - 5 slices')
ax2.set_ylabel('$Q_{y}$')
ax2.set_xlabel('$\delta$')
plt.show()

