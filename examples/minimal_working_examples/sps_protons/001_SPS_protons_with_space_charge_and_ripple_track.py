"""
Minimalistic example to load an xsuite line, generate a particle distribution,
install space charge, ripple and track the particles.

For SPS Q20 protons
"""
import numpy as np
import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo   
from records import Records, get_k_ripple_summed_signal
import time

# Boolean parameters
add_tune_ripple = True
add_non_linear_magnet_errors = True
fname_line = 'sps_q20_proton_line_with_errors.json' if add_non_linear_magnet_errors else 'sps_q20_proton_line.json'

# Tracking parameters and context
n_turns = 20
n_particles = 20
context = xo.ContextCpu(omp_num_threads='auto')
nturns_profile_accumulation_interval = 100 # turn interval between which to aggregate transverse particles for histogram
nbins = 140 # number of bins for histograms of transverse monitors

# Tune ripple parameters
I_QF_QD = 70. # quadrupolar circuit current in ampere used (approx). 70 A used for SPS Pb ions
amp_adjustment_factor_from_current = 70./I_QF_QD

# Transfer function factors, by which we amplitude-adjust the ripple
# From 2018 TBT vs current tune data analysis, by Elias using data from Hannes
# See Elias Waagaard's PhD thesis for details
# For SPS Pb ions, 70 A was used. If different current, adjust amplitudes accordinly
a_50 = 1.0 * amp_adjustment_factor_from_current
a_150 = 0.5098 * amp_adjustment_factor_from_current
a_300 = 0.2360 * amp_adjustment_factor_from_current
a_600 = 0.1095 * amp_adjustment_factor_from_current

# Desired ripple frequencies and amplitudes - typical values without 50 Hz compensation
ripple_freqs = np.array([50.0, 150.0, 300.0, 600.0])
kqf_amplitudes = np.array([1.0141062492337905e-06*a_50, 1.9665396648867768e-07*a_150, 3.1027971430227987e-07*a_300, 4.5102937494506313e-07*a_600])
kqd_amplitudes = np.array([1.0344583265981035e-06*a_50, 4.5225494700433166e-07*a_150, 5.492718035100028e-07*a_300, 4.243698659233664e-07*a_600])
kqf_phases = np.array([0.7646995873548973, 2.3435670020522825, -1.1888958255027886, 2.849205512655574])
kqd_phases = np.array([0.6225130389353318, -1.044380492147742, -1.125401419249802, -0.30971750008702853])

# Typical values with 50 Hz compensation
"""
kqf_amplitudes = np.array([1.6384433351717334e-08*a_50, 2.1158318710898557e-07*a_150, 3.2779826135772383e-07*a_300, 4.7273849059164697e-07*a_600])
kqd_amplitudes = np.array([2.753093584240069e-07*a_50, 4.511100472630622e-07*a_150, 5.796354631307802e-07*a_300, 4.5487568431405856e-07*a_600])
kqf_phases = np.array([0.9192671763874849, 0.030176158557178895, 0.5596488397663701, 0.050511945653341016])
kqd_phases = np.array([0.9985112397758237, 3.003827454851132, 0.6369886405485959, -3.1126209931146547])
"""

# Beam parameters
exn = 1.0e-6  # horizontal emittance in m
eyn = 1.0e-6  # vertical emittance in m
sigma_z = 0.2  # bunch length in m
Nb = 1e11  # number of particles in the bunch

# Space charge parameters
q_val = 1.0  # q-value of longitudinal profile, for space charge
num_spacecharge_interactions = 1080

# Load the line from a file
line = xt.Line.from_json(fname_line)

# Set RF voltage to different value, if desired
#line['actcse.31632' ].voltage = 3.0e6 
#print('RF voltage set to {:.3e} V\n'.format(line['actcse.31632'].voltage))

# Twiss command, inspect tunes and reference particle
tw = line.twiss() 
print('Tunes: Qx = {:.6f}, Qy = {:.6f}'.format(tw.qx, tw.qy))
print('Reference particle: {}'.format(line.particle_ref.show()))

# Create horizontal beam monitor
monitorH = xt.BeamProfileMonitor(
    start_at_turn=nturns_profile_accumulation_interval/2, stop_at_turn=n_turns,
    frev=1,
    sampling_frequency=1/nturns_profile_accumulation_interval,
    n=nbins,
    x_range=0.07,
    y_range=0.07)
line.insert_element(index='bwsrc.51637', element=monitorH, name='monitorH')

# Create vertical beam monitor
monitorV = xt.BeamProfileMonitor(
    start_at_turn=nturns_profile_accumulation_interval/2, stop_at_turn=n_turns,
    frev=1,
    sampling_frequency=1/nturns_profile_accumulation_interval,
    n=nbins,
    x_range=0.07,
    y_range=0.07)
line.insert_element(index='bwsrc.41677', element=monitorV, name='monitorV')



# Build tracker
line.build_tracker(_context=context)

# Generate a particle distribution
particles = xp.generate_matched_gaussian_bunch(_context=context,
    num_particles=n_particles, 
    total_intensity_particles=Nb,
    nemitt_x=exn, 
    nemitt_y=eyn, 
    sigma_z=sigma_z,
    particle_ref=line.particle_ref, 
    line=line)

######### Frozen space charge #########

# Store the initial buffer of the line
_buffer = line._buffer
line.discard_tracker()

# Install space charge
lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles = Nb,
        sigma_z = sigma_z,
        z0=0.,
        q_parameter=q_val)

# Install frozen space charge as base 
xf.install_spacecharge_frozen(line = line,
                    particle_ref = line.particle_ref,
                    longitudinal_profile = lprofile,
                    nemitt_x = exn, nemitt_y = eyn,
                    sigma_z = sigma_z,
                    num_spacecharge_interactions = num_spacecharge_interactions)

# Add tune ripple if desired
if add_tune_ripple:

    # Create ripple in quadrupolar knobs, convert phases to turns
    turns_per_sec = 1/tw['T_rev0']
    ripple_periods = turns_per_sec/ripple_freqs #).astype(int)  # number of turns particle makes during one ripple oscillation
    kqf_phases_turns = kqf_phases * turns_per_sec # convert time domain to turn domain, i.e. multiply with turns/sec
    kqd_phases_turns = kqd_phases * turns_per_sec # convert time domain to turn domain, i.e. multiply with turns/sec

    # Generate custom tune ripple signal
    kqf_ripple, kqd_ripple = get_k_ripple_summed_signal(n_turns, ripple_periods, kqf_amplitudes, kqd_amplitudes,
                                                                    kqf_phases_turns, kqd_phases_turns)
    
    # Save initial values
    kqf0 = line.vars['kqf']._value
    kqd0 = line.vars['kqd']._value
    
    print('Norm. quadrupolar gradients will oscillate with')
    print('kqf =  {:.4e} +/- {:.3e}'.format(kqf0, max(kqf_ripple)))
    print('kqd = {:.4e} +/- {:.3e}'.format(kqd0, max(kqd_ripple)))


# Build tracker
line.build_tracker(_buffer=_buffer)


# Initialize the dataclasses to store particle values
tbt = Records.init_zeroes(n_turns)  # only emittances and bunch intensity
tbt.update_at_turn(0, particles, tw)
tbt.store_initial_particles(particles)
tbt.store_twiss(tw.to_pandas())

# Empty arrays to store data
X_data = np.zeros(n_turns)
Y_data = np.zeros(n_turns)
kqf_data = np.zeros(n_turns)
kqd_data = np.zeros(n_turns)
X_data[0] = np.mean(particles.x)
Y_data[0] = np.mean(particles.y)
kqf_data[0] = line.vars['kqf']._value
kqd_data[0] = line.vars['kqd']._value

#### Track the particles ####
time00 = time.time()

for turn in range(1, n_turns):
    
    # Print out info at specified intervals
    if turn % 5 == 0:
        print('\nTracking turn {}'.format(turn))        
 
    ########## ----- Exert TUNE RIPPLE if desired ----- ##########
    if add_tune_ripple:
        line.vars['kqf'] = kqf0 + kqf_ripple[turn-1]
        line.vars['kqd'] = kqd0 + kqd_ripple[turn-1]

    # ----- Track and update records for tracked particles ----- #
    line.track(particles, num_turns=1)

    # Store centroid and normalized quadrupolar gradient data
    X_data[turn] = np.mean(particles.x)
    Y_data[turn] = np.mean(particles.y)
    kqf_data[turn] = line.vars['kqf']._value
    kqd_data[turn] = line.vars['kqd']._value

    tbt.update_at_turn(turn, particles, tw)

time01 = time.time()
dt0 = time01-time00
print('\nTracking time: {:.1f} s = {:.1f} h'.format(dt0, dt0/3600))

# Convert turns to seconds
turns_per_sec = 1 / tw.T_rev0
num_seconds = n_turns / turns_per_sec # number of seconds we are running for
seconds_array = np.linspace(0.0, num_seconds, num=int(n_turns))

# Final turn-by-turn records
tbt.store_final_particles(particles)
tbt.append_profile_monitor_data(monitorH, monitorV, seconds_array)
tbt.append_centroid_data(X_data, Y_data, kqf_data, kqd_data)
tbt.to_dict(convert_to_numpy=True)

# then save tbt dict if desired