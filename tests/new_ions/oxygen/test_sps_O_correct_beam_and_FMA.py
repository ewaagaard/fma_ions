"""
Check that correct oxygen beam is loaded
"""
import fma_ions

# First, check that reference particle is correct
sps = fma_ions.SPS_sequence_maker(26.30, 26.19, ion_type='O', Q_PS=4., Q_SPS=8., m_ion=15.9949) 
line, twiss = sps.load_xsuite_line_and_twiss()

print('O beam:')
print(line.particle_ref.show())

# Update beam parameters
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = beamParams.Nb_O  # update to new oxygen intensity
print(beamParams)

# Then, check that space charge is installed correctly
sps_fma = fma_ions.FMA()
line = sps_fma.install_SC_and_get_line(line=line, beamParams=beamParams, optimize_for_tracking=False)


        lprofile = xf.LongitudinalProfileQGaussian(
                number_of_particles = beamParams.Nb,
                sigma_z = sigma_z_RMS,
                z0=0.,
                q_parameter=q_val)

        print('\nInstalled SC.')
        print(lprofile)
        print(line.particle_ref.show())
        print(beamParams)

        # Install frozen space charge as base 
        xf.install_spacecharge_frozen(line = line,
                           particle_ref = line.particle_ref,
                           longitudinal_profile = lprofile,
                           nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn,
                           sigma_z = sigma_z_RMS,
                           num_spacecharge_interactions = self.num_spacecharge_interactions)

# Test default tracking with space charge on CPU context - then test plotting
#sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=100, num_turns=100, turn_print_interval=10)
#tbt = sps.track_SPS(ion_type='O', which_context='cpu', add_aperture=True, apply_kinetic_IBS_kicks=True)
#sps.plot_tracking_data(tbt, show_plot=True)
