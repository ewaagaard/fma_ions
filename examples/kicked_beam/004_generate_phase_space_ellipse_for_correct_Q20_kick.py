"""
Example to reproduce external kick given to Q20 proton experiments in Oct/Nov 2018
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import xobjects as xo

def transfer_matrix(beta_B, alpha_B, beta_A, alpha_A, delta_mu):
    """
    Compute the transfer matrix between two locations in a beamline
    using the Twiss parameters and the phase advance.
    
    Parameters:
    beta_B : float  -> Beta function at location B (start)
    alpha_B : float -> Alpha function at location B (start)
    beta_A : float  -> Beta function at location A (WS)
    alpha_A : float -> Alpha function at location A (WS)
    delta_mu : float -> Phase advance (in radians) from B to A
    
    Returns:
    numpy.ndarray -> 2x2 transfer matrix
    """
    R11 = np.sqrt(beta_A / beta_B) * (np.cos(delta_mu) + alpha_B * np.sin(delta_mu))
    R12 = np.sqrt(beta_A * beta_B) * np.sin(delta_mu)
    R21 = (alpha_A - alpha_B) / np.sqrt(beta_A * beta_B) * np.cos(delta_mu) - (1 + alpha_A * alpha_B) / np.sqrt(beta_A * beta_B) * np.sin(delta_mu)
    R22 = np.sqrt(beta_B / beta_A) * (np.cos(delta_mu) - alpha_A * np.sin(delta_mu))
    
    return np.array([[R11, R12], [R21, R22]])

# Define parameters and desired amplitudes
X_amp_at_BWS = 1e-3

# Generate the line
sps_seq = fma_ions.SPS_sequence_maker()
line, twiss_sps = sps_seq.load_xsuite_line_and_twiss(add_aperture=True)

# Find the beta function and phase advance
df = twiss_sps.to_pandas()
betx0 = df.iloc[0].betx
betx = df[df['name'] == 'bwsrc.51637'].betx.values[0]
alfx0 = df.iloc[0].alfx
alfx = df[df['name'] == 'bwsrc.51637'].alfx.values[0]
mux = df[df['name'] == 'bwsrc.51637'].mux.values[0]
kick = X_amp_at_BWS / (np.sqrt(betx*betx0) * np.sin(mux))

# Also compute the values 
R = transfer_matrix(betx0, alfx0, betx, alfx, mux)
V1 = np.array([0., kick])
V2 = R.dot(V1)
print('X vector at WS location with initial computed kick: {}\n'.format(V2))

# Generate particles to track, kick a copy of particles
num_turns = 100
num_particles = 100
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part = num_particles, num_turns = num_turns)
particles = sps.generate_particles(line)
particles0 = particles.copy()
#particles2 = particles.copy()
#particles.x -= np.mean(particles.x)
particles.px += kick
#particles2.x += 1e-3

# Install a beam position monitor to check
monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=num_turns-1,
                                    num_particles=num_particles)
monitor2 = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=num_turns-1,
                                    num_particles=num_particles)
line.unfreeze() # if you had already build the tracker
line.insert_element(index='bwsrc.51637', element=monitor, name='mymon')
line.insert_element(index='bwsrc.51637', element=monitor2, name='mymon2')
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

mean = []
mean2 = []

for iturn in range(num_turns):
    mean.append(np.mean(particles.x))
    #mean2.append(np.mean(particles2.x))
    monitor.track(particles)
    #monitor2.track(particles2)
    line.track(particles)
    #line.track(particles2)
    if iturn % 10 == 0:
        print(f'Turn {iturn}')
        


# Plot the centroids
fig0, ax0 = plt.subplots(1, 1, figsize=(8,6))
ax0.plot(mean, color='b', marker='o', alpha=0.8)
#ax0.plot(mean2, color='r', marker='v', alpha=0.75)

fig00, ax00 = plt.subplots(1, 1, figsize=(8,6))
ax00.plot(np.mean(monitor.x, axis=0), color='b', marker='o', alpha=0.8)
ax00.set_ylabel("X")
ax00.set_xlabel("Turn")
#ax00.plot(np.mean(monitor2.x, axis=0), color='r', marker='v', alpha=0.75)
    
# Plot the resulting phase space
fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.plot(monitor.x, monitor.px, color='b', ls='None', marker='o', alpha=0.8)
ax.plot(monitor2.x, monitor2.px, color='r', ls='None', marker='v', alpha=0.75)
ax.set_ylabel("X'")
ax.set_xlabel("X")
plt.show()