"""
Simple test script to launch SPS tracking with/without longitudinal beam monitor to see speed
"""
import fma_ions
import time

n_turns = 200
num_part = 10_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part, turn_print_interval=20)

# First track without beam monitor
time00 = time.time()
tbt = sps.track_SPS(which_context='cpu', install_SC_on_line=False, add_aperture=True, install_beam_monitors=False)
time01 = time.time()
dt0 = time01-time00

# Then track with beam monitor
time10 = time.time()
tbt2 = sps.track_SPS(which_context='cpu', install_SC_on_line=False, add_aperture=True, install_beam_monitors=True)
time11 = time.time()
dt1 = time11-time10
print('\nTracking time: \nwithout monitors t0 = {:.4f} s \nwith monitors t1 = {:.4f} s'.format(dt0, dt1))