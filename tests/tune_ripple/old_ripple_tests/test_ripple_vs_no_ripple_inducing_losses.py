"""
Small test script to investigate whether tunes are changed as expected
- include beta-beat and magnet errors and space charge
Plot particle losses
"""
import numpy as np
import fma_ions
import matplotlib.pyplot as plt
import xtrack as xt
import time

num_turns = 200
turn_range = np.arange(100)
num_part = 200
ripple_freq = 500 # Hz
beamParams = fma_ions.BeamParameters_SPS

# Load line
sps = fma_ions.SPS_sequence_maker(qx0=26.30, qy0=26.13)
line, twiss = sps.load_xsuite_line_and_twiss(add_aperture=True, add_non_linear_magnet_errors=True,
                                              beta_beat=0.15)
sps.set_LSE_sextupolar_errors(line)
sps.set_LOE_octupolar_errors(line)
line.build_tracker()
twiss = line.twiss()

# Generate particles
sps_tracker = fma_ions.SPS_Flat_Bottom_Tracker(num_part=num_part, num_turns=num_turns)
particles = sps_tracker.generate_particles(line=line)
fma_sps = fma_ions.FMA()

'''
# Generate extra particles - test without tune ripple
particles2 = particles.copy()
line2 = line.copy()
line2.build_tracker()

# Install space charge elements on both lines
line = fma_sps.install_SC_and_get_line(line=line,
                                        beamParams=beamParams,
                                        optimize_for_tracking=False)
line2 = fma_sps.install_SC_and_get_line(line=line2,
                                        beamParams=beamParams,
                                        optimize_for_tracking=False)
'''

# Set upp tune ripple
turns_per_sec = 1/twiss['T_rev0']
ripple_period = int(turns_per_sec/ripple_freq)  # number of turns particle makes during one ripple oscillation
ripple = fma_ions.Tune_Ripple_SPS(num_turns=num_turns, ripple_period=ripple_period, qx0=sps_tracker.qx0, qy0=sps_tracker.qy0)
kqf_vals, kqd_vals, _ = ripple.load_k_from_xtrack_matching(dq=0.01, plane='both')


# Track with tune ripple
for turn in range(num_turns):
    print('\nRipple: Tracking turn {}'.format(turn))      
    print('kqf = {:.7f}, kqf = {:.7f}'.format(line.vars['kqf']._value, line.vars['kqd']._value))

    # First change knobs for tune ripple
    line.vars['kqf'] = kqf_vals[turn]
    line.vars['kqd'] = kqd_vals[turn]
    
    # Track the particle object
    line.track(particles, num_turns=1)
    if particles.state[particles.state <= 0].size > 0:
        print('Lost particle state: most common code: "-{}" for {} particles out of {} lost in total'.format(np.bincount(np.abs(particles.state[particles.state <= 0])).argmax(),
                                                                                            np.max(np.bincount(np.abs(particles.state[particles.state <= 0]))),
                                                                                            len(particles.state[particles.state <= 0])))
'''    
# Track without tune ripple
for turn in range(num_turns):
    print('\nNo ripple: Tracking turn {}'.format(turn))      
    
    # Track the particle object
    line2.track(particles2, num_turns=1)
    if particles2.state[particles2.state <= 0].size > 0:
        print('Lost particle state: most common code: "-{}" for {} particles out of {} lost in total'.format(np.bincount(np.abs(particles2.state[particles.state <= 0])).argmax(),
                                                                                            np.max(np.bincount(np.abs(particles2.state[particles2.state <= 0]))),
                                                                                            len(particles2.state[particles2.state <= 0])))
'''
print('Summary:')
print('Ripple: lost particle state: most common code: "-{}" for {} particles out of {} lost in total'.format(np.bincount(np.abs(particles.state[particles.state <= 0])).argmax(),
                                                                                        np.max(np.bincount(np.abs(particles.state[particles.state <= 0]))),
                                                                                        len(particles.state[particles.state <= 0])))
#print('No ripple: lost particle state: most common code: "-{}" for {} particles out of {} lost in total'.format(np.bincount(np.abs(particles2.state[particles.state <= 0])).argmax(),
#                                                                                        np.max(np.bincount(np.abs(particles2.state[particles2.state <= 0]))),
#                                                                                        len(particles2.state[particles2.state <= 0])))