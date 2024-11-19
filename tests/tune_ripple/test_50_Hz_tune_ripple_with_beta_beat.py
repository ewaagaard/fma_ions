"""
Small test script to investigate whether tunes are changed as expected
- include beta-beat and magnet errors
"""
import numpy as np
import fma_ions
import matplotlib.pyplot as plt
import xtrack as xt
import time

num_turns = 100
turn_range = np.arange(100)
num_part = 30
ripple_freq = 500 # Hz

# Load line
sps = fma_ions.SPS_sequence_maker()
line = sps.generate_xsuite_seq_with_beta_beat(add_aperture=True, add_non_linear_magnet_errors=True,
                                              beta_beat=0.01)
sps.set_LSE_sextupolar_errors(line)
sps.set_LOE_octupolar_errors(line)
line.build_tracker()
twiss = line.twiss()

# Generate particles
sps_tracker = fma_ions.SPS_Flat_Bottom_Tracker(num_part=num_part, num_turns=num_turns)
particles = sps_tracker.generate_particles(line=line)

# Set upp tune ripple
turns_per_sec = 1/twiss['T_rev0']
ripple_period = int(turns_per_sec/ripple_freq)  # number of turns particle makes during one ripple oscillation
ripple = fma_ions.Tune_Ripple_SPS(num_turns=num_turns, ripple_period=ripple_period, qx0=sps_tracker.qx0, qy0=sps_tracker.qy0)
kqf_vals, kqd_vals, _ = ripple.load_k_from_xtrack_matching(dq=0.1, plane='both')

qx = np.zeros(num_turns)
qy = np.zeros(num_turns)

# Track
time00 = time.time()
for turn in range(num_turns):
    print('\nTracking turn {}'.format(turn))      
    
    # First change knobs for tune ripple
    line.vars['kqf'] = kqf_vals[turn]
    line.vars['kqd'] = kqd_vals[turn]
    
    tw = line.twiss()
    qx[turn] = tw['qx']
    qy[turn] = tw['qy']
    
    # Track the particle object
    line.track(particles, num_turns=1)
    
# Plot tunes over time
fig, ax = plt.subplots(1,1, figsize=(8,6), constrained_layout=True)
ax.plot(turn_range, qx, label='Qx')
ax.plot(turn_range, qy, label='Qy')
ax.set_xlabel('Turn')
ax.set_ylabel('Tune')
ax.legend()
plt.show()

# Check that aperture actually is installed
line.check_aperture()
x_ap, y_ap, a = sps.print_smallest_aperture(line)