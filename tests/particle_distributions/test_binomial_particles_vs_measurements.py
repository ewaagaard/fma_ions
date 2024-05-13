"""
Generate binomial particle distribution vs measured profiles at PS extraction and SPS injection
"""
import numpy as np
import matplotlib.pyplot as plt
import fma_ions
import xobjects as xo

# Generate SPS sequence
sps_maker = fma_ions.SPS_sequence_maker()
line, twiss = sps_maker.load_xsuite_line_and_twiss()
context = xo.ContextCpu()

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=500_000, num_turns=100, turn_print_interval=10)
particles = sps.generate_particles(line, context, distribution_type='binomial')


tbt = fma_ions.Full_Records.init_zeroes(len(particles.x[particles.state > 0]), 1,
                                        which_context='cpu', full_data_turn_ind=[0]) # full particle data
tbt.update_at_turn(0, particles, twiss)
sps.compare_longitudinal_phase_space_vs_data(tbt_dict=tbt, include_final_turn=False, num_bins=100)
