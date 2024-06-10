"""
SPS lattice with Pb ions - GPU
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 2_000_000
num_part = 10_000

# Instantiate beam parameters, custom made to compare with 2016 measurements
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = beamParams.Nb * 0.95  # loss factor from first turn observed with wall current monitor
beamParams.sigma_z_binomial = 0.225 # what we measure after initial losses out of the RF bucket
beamParams.m = 2.8 # meausred for binomial after SPS injection

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', beamParams=beamParams, install_SC_on_line=True, distribution_type='binomial')
tbt.to_json(output_dir)
