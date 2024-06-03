"""
Generate binomial particle distribution vs measured profiles at SPS injection, but after some particles have been spilled out
"""
import numpy as np
import matplotlib.pyplot as plt
import fma_ions
import xobjects as xo

# Generate SPS sequence
sps_maker = fma_ions.SPS_sequence_maker()
line, twiss = sps_maker.load_xsuite_line_and_twiss()
context = xo.ContextCpu()

# Update bunch length
beamParams = fma_ions.BeamParameters_SPS()
beamParams.sigma_z_binomial = 0.215 # more accurate starting value after RF spill
m=2.8 # meausred for binomial after SPS injection

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=500_000, num_turns=100)
particles = sps.generate_particles(line, context, distribution_type='binomial', beamParams=beamParams, m=m)

# initialize first particle data
tbt = fma_ions.Full_Records.init_zeroes(len(particles.x[particles.state > 0]), 1,
                                        which_context='cpu', full_data_turn_ind=[0]) # full particle data
tbt.update_at_turn(0, particles, twiss)
sps.compare_longitudinal_phase_space_vs_data(tbt_dict=tbt, include_final_turn=False, num_bins=80,
                                             generate_new_zero_turn_binomial_particle_data_without_pretracking=False,
                                             also_show_SPS_inj_profile_after_RF_spill=True)
