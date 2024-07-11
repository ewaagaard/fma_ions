"""
Script to track particles --> compare bunch length of particles vs profile monitor 
"""
import fma_ions
import time
import matplotlib
import numpy as np
import xpart as xp

num_part = 10_000
num_turns = 500
zeta_container_interval = 100

# SPS sequence generator
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()

# Import beam parameters, and generate Gaussian beam
beamParams = fma_ions.BeamParameters_SPS_Binomial_2016()
zeta, delta = xp.longitudinal.generate_longitudinal_coordinates(line=line, distribution='qgaussian',
                                                                num_particles=num_part,
                                                                engine='single-rf-harmonic', sigma_z=beamParams.sigma_z,
                                                                particle_ref=line.particle_ref, return_matcher=False, q=beamParams.q)
# Initiate normalized coordinates
x_norm = np.random.normal(size=num_part)
px_norm = np.random.normal(size=num_part)
y_norm = np.random.normal(size=num_part)
py_norm = np.random.normal(size=num_part)

particles = xp.build_particles(_context=None, particle_ref=line.particle_ref, 
                                zeta=zeta, delta=delta,
                                x_norm=x_norm, px_norm=px_norm,
                                y_norm=y_norm, py_norm=py_norm,
                                nemitt_x=beamParams.exn, nemitt_y=beamParams.eyn,
                                weight=beamParams.Nb/num_part, line=line)


# Initialize a zeta container (to contain particles every 100 turns)
zetas = fma_ions.Zeta_Container.init_zeroes(len(particles.x), 100, which_context='cpu')
zetas.update_at_turn(0, particles)

# Array with bunch length and TBT monitor
BL = []
tbt = fma_ions.Records.init_zeroes(num_turns)  # only emittances and bunch intensity
tbt.update_at_turn(0, particles, twiss)

# Start tracking 
time00 = time.time()
for turn in range(1, num_turns):
    print('\nTracking turn {}'.format(turn))   

    # Track particles and fill zeta container
    line.track(particles, num_turns=1)
    tbt.update_at_turn(turn, particles, twiss)
    zetas.update_at_turn(turn % zeta_container_interval, particles) 

    # Merge all longitudinal coordinates to profile and stack
    if (turn+1) % zeta_container_interval == 0:
                
        # Aggregate longitudinal coordinates of particles still alive - into bunch length
        z = zetas.zeta.flatten()
        s = zetas.state.flatten()
        zetas_accumulated = z[s>0]
        BL.append(np.std(zetas_accumulated))
        print('Turn {} - calculating aggregated bunch length'.format(turn))

        # Initialize new zeta containers
        del zetas
        zetas = fma_ions.Zeta_Container.init_zeroes(len(particles.x), zeta_container_interval, 
                                    which_context='cpu')
        zetas.update_at_turn(0, particles) # start from turn, but 0 in new dataclass
                
time01 = time.time()
dt0 = time01-time00
print('\nTracking time: {:.1f} s = {:.1f} h'.format(dt0, dt0/3600))

# Make array with turns
turn_array = np.arange(0, num_turns, step=zeta_container_interval)

        f3, ax22 = plt.subplots(1, 1, figsize = (8,6))
        # Uncomment if want to plot standard deviation of numerical particle object
        if also_plot_particle_std_BL:
            ax22.plot(time_units, tbt_dict['bunch_length'], color='darkcyan', alpha=0.7, lw=1.5, label='STD($\zeta$) of simulated particles')      
            ax22.plot(turn_array if x_unit_in_turns else time_array, sigmas_gaussian, color='cyan', ls='--', alpha=0.95,
                      label='Simulated profiles')
