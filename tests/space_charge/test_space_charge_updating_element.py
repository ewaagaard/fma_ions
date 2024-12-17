"""
For rapidly changing beam parameters, update space charge legnths and beam sizes
"""
import numpy as np
import xfields as xf
import xpart as xp
import fma_ions
import time
import matplotlib.pyplot as plt

# Beam parameters for the tracking
num_turns = 100
num_part = 100
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = 3.5e10 # exaggerate bunch intensity
update_sc_interval = 10

# Load line from sequence generator
sps_seq = fma_ions.SPS_sequence_maker(qy0=26.10)
line, twiss = sps_seq.load_xsuite_line_and_twiss(add_aperture=True)

# Add beta-beat
line.element_refs['qd.63510..1'].knl[1] = -1.07328640311457e-02
line.element_refs['qf.63410..1'].knl[1] = 1.08678014669101e-02
print('Beta-beat added: kk_QD = {:.6e}, kk_QF = {:.6e}'.format(line.element_refs['qd.63510..1'].knl[1]._value,
                                                                line.element_refs['qf.63410..1'].knl[1]._value))

# Find particles and initial intensity
particles = xp.generate_matched_gaussian_bunch(num_particles=num_part, 
    total_intensity_particles=beamParams.Nb,
    nemitt_x=beamParams.exn, 
    nemitt_y=beamParams.eyn, 
    sigma_z= beamParams.sigma_z,
    particle_ref=line.particle_ref, 
    line=line)
Nb0 = particles.weight[particles.state > 0][0]*len(particles.x[particles.state > 0]) # initial bunch intensity


# Install frozen space charge
lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles = beamParams.Nb,
        sigma_z = beamParams.sigma_z,
        z0=0.)

xf.install_spacecharge_frozen(line = line,
                    particle_ref = line.particle_ref,
                    longitudinal_profile = lprofile,
                    nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn,
                    sigma_z = beamParams.sigma_z,
                    num_spacecharge_interactions = 1080)

line.build_tracker()

# Space charge elements will have similar length
ee0_elements = []
ee0_element_lengths = []
for ii, ee in enumerate(line.elements):
    if isinstance(ee, xf.SpaceChargeBiGaussian):
        ee0_elements.append(ee)
        ee0_element_lengths.append(ee.length)
print('Initial SC element lengths = {:.5f} m +- {:.3e}'.format(np.mean(ee0_element_lengths), np.std(ee0_element_lengths)))
ee0_length = ee0_element_lengths[0]

# Bunch intensity
Nb_vals = []

# Track particles
time00 = time.time()
for turn in range(0, num_turns):

    Nb = particles.weight[particles.state > 0][0]*len(particles.x[particles.state > 0])
    transmission = Nb/Nb0
    Nb_vals.append(Nb)

    if turn % update_sc_interval == 0:
        for ii, ee in enumerate(line.elements):
            if isinstance(ee, xf.SpaceChargeBiGaussian):
                
                # Scale length with bunch intensity
                ee.length = transmission*ee0_length
        print('Turn = {} - re-adjusted SC element length by {:.4f}'.format(turn, transmission))

    line.track(particles, num_turns=1)


time01 = time.time()
dt0 = time01-time00
print('\nTracking time: {:.4f} s h'.format(dt0))

ee_element_lengths = []
for ii, ee in enumerate(line.elements):
    if isinstance(ee, xf.SpaceChargeBiGaussian):
        ee_element_lengths.append(ee.length)
print('Final SC element lengths = {:.5f} m +- {:.3e}'.format(np.mean(ee_element_lengths), np.std(ee_element_lengths)))
ee0_length = ee0_element_lengths[0]

# Plot result space charge element length, and bunch intensity
turns = np.arange(1, num_turns)
fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex=True)
ax[0].plot(turns, Nb_vals)
ax[0].set_ylabel('Nb')
ax[1].plot(turns, np.array(Nb_vals)/Nb0*ee0_length)
ax[1].set_ylabel('SC ele. length [m]')
ax[1].set_xlabel('Turns')
plt.show()