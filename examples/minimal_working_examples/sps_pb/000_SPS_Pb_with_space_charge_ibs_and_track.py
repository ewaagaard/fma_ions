"""
Minimalistic example to load an xsuite line, generate a particle distribution,
install space charge, IBS, and track the particles.
"""
import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo   
from records import Records
import time

# Tracking parameters and context
n_turns = 20
n_particles = 20
context = xo.ContextCpu(omp_num_threads='auto')

# Beam parameters
exn = 1.0e-6  # horizontal emittance in m
eyn = 1.0e-6  # vertical emittance in m
sigma_z = 0.2  # bunch length in m
Nb = 1e8  # number of particles in the bunch

# Space charge and IBS parameters
q_val = 1.0  # q-value of longitudinal profile, for space charge
num_spacecharge_interactions = 1080
ibs_step = 100  # number of turns between recomputing the IBS coefficients

# Load the line from a file
line = xt.Line.from_json('SPS_2021_Pb_nominal_deferred_exp.json')
harmonic_nb = 4653

# Set RF voltage to correct value
line['actcse.31632' ].voltage = 3.0e6 
print('RF voltage set to {:.3e} V\n'.format(line['actcse.31632'].voltage))

# Twiss command, inspect tunes and reference particle
tw = line.twiss() 
print('Tunes: Qx = {:.6f}, Qy = {:.6f}'.format(tw.qx, tw.qy))
print('Reference particle: {}'.format(line.particle_ref.show()))

# Add longitudinal limit rectangle - to kill particles that fall out of bucket
bucket_length = line.get_length()/harmonic_nb
print('\nBucket length is {:.4f} m'.format(bucket_length))
line.unfreeze() # if you had already build the tracker
line.append_element(element=xt.LongitudinalLimitRect(min_zeta=-bucket_length/2, max_zeta=bucket_length/2), name='long_limit')
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

# Initialize the dataclasses to store particle values
tbt = Records.init_zeroes(n_turns)  # only emittances and bunch intensity
tbt.update_at_turn(0, particles, tw)
tbt.store_initial_particles(particles)
tbt.store_twiss(tw.to_pandas())

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
line.build_tracker(_buffer=_buffer)

######### IBS kinetic kicks #########

#  friction and diffusion terms of the kinetic theory of gases
ibs_kick = xf.IBSKineticKick(num_slices=50)

### Install the IBS kinetic kick element ###
#line.configure_intrabeam_scattering(
#    element=ibs_kick, name="ibskick", index=-1, update_every=ibs_step
#)

# THESE LINES ABOVE WILL NOT WORK if space charge is already installed
# Instead, follow manual steps Felix Soubelet's tips
# Directly copy steps from https://github.com/xsuite/xfields/blob/6882e0d03bb6772f873ce57ef6cf2592e5779359/xfields/ibs/_api.py
_buffer = line._buffer
line.discard_tracker()
line.insert_element(element=ibs_kick, name="ibskick", index=-1)
line.build_tracker(_buffer=_buffer)

line_sc_off = line.filter_elements(exclude_types_starting_with='SpaceCh')
twiss_no_sc = line_sc_off.twiss(method="4d")

# Figure out the IBS kick element and its name in the line
only_ibs_kicks = {name: element for name, element in line.element_dict.items() if isinstance(element, xf.ibs._kicks.IBSKick)}
assert len(only_ibs_kicks) == 1, "Only one 'IBSKick' element should be present in the line"
name, element = only_ibs_kicks.popitem()

# Set necessary (private) attributes for the kick to function
element.update_every = ibs_step
element._name = name
element._twiss = twiss_no_sc
element._scale_strength = 1  # element is now ON, will track

print('\nFixed IBS coefficient recomputation at interval = {} steps\n'.format(ibs_step))

#### Track the particles ####
time00 = time.time()

for turn in range(1, n_turns):
    
    # Print out info at specified intervals
    if turn % 5 == 0:
        print('\nTracking turn {}'.format(turn))        

    # ----- Track and update records for tracked particles ----- #
    line.track(particles, num_turns=1)

    tbt.update_at_turn(turn, particles, tw)

time01 = time.time()
dt0 = time01-time00
print('\nTracking time: {:.1f} s = {:.1f} h'.format(dt0, dt0/3600))

# Final turn-by-turn records
tbt.store_final_particles(particles)
tbt.to_dict(convert_to_numpy=True)

# then save tbt dict if desired