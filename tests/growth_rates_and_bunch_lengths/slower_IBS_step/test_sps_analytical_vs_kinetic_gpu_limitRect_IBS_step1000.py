"""
Test tracking with kinetic IBS kicks vs Nagaitsev formalism, for SPS set-up - with GPU for longer
Slower IBS updates
(Qx, Qy) = (26.30, 26.19) - ideal lattice
Include longitudinalLimitRect
"""
import fma_ions
import xobjects as xo

# Define cupy context
context = xo.ContextCupy()

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=5000, num_turns=10_000, turn_print_interval=100)

beamParams2 = fma_ions.BeamParameters_SPS()
beamParams2.sigma_z = 0.19

# First test 
sps.run_analytical_vs_kinetic_emittance_evolution(Qy_frac=19, which_context='gpu', ibs_step=1500,
                                                  beamParams=beamParams2, context=context, install_longitudinal_rect=True,
                                                  extra_plot_string='_longitudinalLimitRect_ideal_lattice_sigmaZ_0dot19')