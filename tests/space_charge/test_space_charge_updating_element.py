"""
For rapidly changing beam parameters, update space charge legnths and beam sizes
"""
import numpy as np
import xfields as xf
import xpart as xp
import xtrack as xt

import fma_ions
import time
import matplotlib.pyplot as plt

# Beam parameters for the tracking
num_turns = 300
num_part = 200
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = 3.5e10 # exaggerate bunch intensity
dpp = 1e-3
update_sc_interval = 10
nturns_profile_accumulation_interval = 10
nbins = 40

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
line00 = line.copy()

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

# Copy line, replace collective elements with markers for stable twiss
line0 = line.copy()
for ii, key in enumerate(line0.element_names):
    if 'spacecharge' in key:
        line0.element_dict[key] = xt.Marker()
line0.build_tracker()
tw = line0.twiss()
df_twiss = tw.to_pandas()

# Space charge elements will have similar length
s_coord = line.get_s_elements()
ee0_elements = []
ee0_element_lengths = []
ee0_sigma_x = []
ee0_sigma_x_betatronic = []
ee0_sigma_y = []
ee0_s = []
for ii, ee in enumerate(line.elements):
    if isinstance(ee, xf.SpaceChargeBiGaussian):
        ee0_elements.append(ee)
        ee0_element_lengths.append(ee.length)
        ee0_s.append(s_coord[ii])

        # Find beam sizes, do not remove dispersive components
        dx = df_twiss.iloc[np.abs(df_twiss['s'] - s_coord[ii]).argmin()].dx
        ee0_sigma_x_betatronic.append(np.sqrt((ee.sigma_x)**2 - (dpp * dx)**2))
        ee0_sigma_x.append(ee.sigma_x)
        ee0_sigma_y.append(ee.sigma_y)
        
print('Initial SC element lengths = {:.4e} m +- {:.3e}'.format(np.mean(ee0_element_lengths), np.std(ee0_element_lengths)))
print('Initial SC element sigma_x = {:.4e} m +- {:.3e}'.format(np.mean(ee0_sigma_x), np.std(ee0_sigma_x)))
print('Initial SC element sigma_y = {:.4e} m +- {:.3e}'.format(np.mean(ee0_sigma_y), np.std(ee0_sigma_y)))
ee0_length = ee0_element_lengths[0]

# Normalized beam sizes - find them for each SC element, use beta functions closest to these locations
betx_sc = np.zeros(len(ee0_s))
bety_sc = np.zeros(len(ee0_s))
for ii, s in enumerate(ee0_s):
    betx_sc[ii] = df_twiss.iloc[np.abs(df_twiss['s'] - s).argmin()].betx
    bety_sc[ii] = df_twiss.iloc[np.abs(df_twiss['s'] - s).argmin()].bety

#### SPACE CHARGE sigma update, if desired ####
# Also insert beam profile monitors at the start, at location s = 0 with lowest dispersion
# Both vertical and horizontal plane
monitor0 = xt.BeamProfileMonitor(
    start_at_turn=nturns_profile_accumulation_interval/2, stop_at_turn=num_turns,
    frev=1,
    sampling_frequency=1/nturns_profile_accumulation_interval,
    n=nbins,
    x_range=0.07,
    y_range=0.07)
line.insert_element(at=0, element=monitor0, name='monitorH_0')
line.build_tracker()

# Initiate fit functions
fits = fma_ions.Fit_Functions()

# Track particles
Nb_vals = []
time00 = time.time()
monitor_counter = 0

for turn in range(0, num_turns):

    Nb = particles.weight[particles.state > 0][0]*len(particles.x[particles.state > 0])
    transmission = Nb/Nb0
    Nb_vals.append(Nb)

    # Update space charge parameters
    if turn > 0 and turn % update_sc_interval == 0:

        # Fit Gaussian beam sizes to beam profile data
        try:
            ###  Fit beam sizes ###
            popt_X, pcov_X = fits.fit_Gaussian(monitor0.x_grid, monitor0.x_intensity[monitor_counter] / np.max(monitor0.x_intensity[monitor_counter]), p0=(1.0, 0.0, 0.02))
            popt_Y, pcov_Y = fits.fit_Gaussian(monitor0.y_grid, monitor0.y_intensity[monitor_counter] / np.max(monitor0.y_intensity[monitor_counter]), p0=(1.0, 0.0, 0.02))
            monitor_counter += 1
            sigma_raw_X = np.abs(popt_X[2])
            sigma_raw_Y = np.abs(popt_Y[2])
            sigma_norm_X = sigma_raw_X / np.sqrt(df_twiss.betx[0])
            sigma_norm_Y = sigma_raw_Y / np.sqrt(df_twiss.bety[0])

            ### Update space charge element sigmas ###
            sigma_X_sc_elements = sigma_norm_X * np.sqrt(betx_sc)
            sigma_Y_sc_elements = sigma_norm_Y * np.sqrt(bety_sc)
            
            sc_element_counter = 0
            for ii, ee in enumerate(line.elements):
                if isinstance(ee, xf.SpaceChargeBiGaussian):
                    
                    # Scale length with bunch intensity
                    ee.sigma_x = sigma_X_sc_elements[sc_element_counter]
                    ee.sigma_y = sigma_Y_sc_elements[sc_element_counter]
                    sc_element_counter += 1


        except ValueError:
            print('Could not fit beam profiles!')

        # Update SC element lengths
        for ii, ee in enumerate(line.elements):
            if isinstance(ee, xf.SpaceChargeBiGaussian):
                
                # Scale length with bunch intensity
                ee.length = transmission*ee0_length
        print('Turn = {} - re-adjusted first SC element\n --> length by {:.4f}\nFirst SC element beam sizes:\nsigma_x = {:.5f}m \nsigma_y = {:.5f} m\n'.format(turn, transmission,
                                                                                                                              sigma_X_sc_elements[0],
                                                                                                                               sigma_Y_sc_elements[0]))


    line.track(particles, num_turns=1)


time01 = time.time()
dt0 = time01-time00
print('\nTracking time: {:.4f} s h'.format(dt0))

ee_element_lengths = []
ee_sigma_x = []
ee_sigma_y = []
for ii, ee in enumerate(line.elements):
    if isinstance(ee, xf.SpaceChargeBiGaussian):
        ee_element_lengths.append(ee.length)
        ee_sigma_x.append(ee.sigma_x)
        ee_sigma_y.append(ee.sigma_y)
print('Final SC element lengths = {:.5f} m +- {:.3e}'.format(np.mean(ee_element_lengths), np.std(ee_element_lengths)))
print('Final SC element sigma_x = {:.5e} m +- {:.3e}'.format(np.mean(ee_sigma_x), np.std(ee_sigma_x)))
print('Final SC element sigma_y = {:.5e} m +- {:.3e}'.format(np.mean(ee_sigma_y), np.std(ee_sigma_y)))

# Plot result space charge element length, and bunch intensity
turns = np.arange(num_turns)
fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex=True, constrained_layout=True)
ax[0].plot(turns, Nb_vals)
ax[0].set_ylabel('Nb')
ax[1].plot(turns, np.array(Nb_vals)/Nb0*ee0_length)
ax[1].set_ylabel('SC ele. length [m]')
ax[1].set_xlabel('Turns')

# Plot sigma_x and sigma_y at SC element locations
fig2, ax2 = plt.subplots(2, 1, figsize=(9.5,6), sharex=True, constrained_layout=True)
ax2[0].plot(ee0_s, ee0_sigma_x, color='blue', label='$\sigma_{x}$, initial')
ax2[0].set_ylabel('$\sigma_{x}$ [m]')
#ax2[1].plot(ee0_s, ee0_sigma_x_betatronic, color='cyan')
#ax2[1].set_ylabel('$\sigma_{x}$ betatronic [m]')
ax2[1].plot(ee0_s, ee0_sigma_y, color='red', label='$\sigma_{y}$, initial')
ax2[1].set_ylabel('$\sigma_{y}$ [m]')
ax2[1].set_xlabel('s [m]')
for a in ax2:
    a.legend(fontsize=10)

fig3, ax3 = plt.subplots(2, 1, figsize=(8,6), sharex=True, constrained_layout=True)
ax3[0].plot(ee0_s, np.array(ee_sigma_x), color='blue', label='$\sigma_{x}$, final')
ax3[0].plot(ee0_s, np.array(ee_sigma_x) / np.sqrt(betx_sc), color='cyan', label='Norm. $\sigma_{x}$, final')
ax3[0].set_ylabel('$\\bar{\sigma_{x}}$ [m]')
ax3[1].plot(ee0_s,  np.array(ee_sigma_y), color='red', label='$\sigma_{y}$, final')
ax3[1].plot(ee0_s,  np.array(ee_sigma_y) / np.sqrt(bety_sc), color='darkred', label='Norm. $\sigma_{y}$, final')
ax3[1].set_ylabel('$\\bar{\sigma_{y}}$ [m]')
ax3[1].set_xlabel('s [m]')
for a in ax3:
    a.legend(fontsize=10)

plt.show()