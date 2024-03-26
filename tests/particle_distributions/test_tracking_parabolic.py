"""
Test tracking longitudinally parabolic distribution with kinetic IBS kicks applied and tune ripple
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100, turn_print_interval=10)

# Update parabolic RMS bunch length
beamParams = fma_ions.BeamParameters_SPS
beamParams.sigma_z = 0.284

tbt = sps.track_SPS(which_context='cpu', distribution_type='parabolic', beamParams=beamParams,
                    add_aperture=True, apply_kinetic_IBS_kicks=True, add_tune_ripple=True)

#### HAVE TO CHECK FROZEN SPACE CHARGE - SEEMS LIKE ONLY Q-GAUSSIAN AND COASTING ARE IMPLEMENTED