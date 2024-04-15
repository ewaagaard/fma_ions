"""
Check that correct oxygen beam is loaded
"""
import fma_ions
import xfields as xf
import xtrack as xt
import xpart as xp
import xobjects as xo
import os

# Initiate beam parameters
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb_O = 1e9
num_part = 10_000
which_context='gpu'

output_folder = 'output_fma_lower_intensity'
os.makedirs(output_folder, exist_ok=True)

# Select correct context
if which_context=='gpu':
    context = xo.ContextCupy()
elif which_context=='cpu':
    context = xo.ContextCpu(omp_num_threads='auto')

# First, check that reference particle is correct
sps = fma_ions.SPS_sequence_maker(26.30, 26.19, ion_type='O', Q_PS=4., Q_SPS=8., m_ion=15.9949) 
line, twiss_sps = sps.load_xsuite_line_and_twiss()
line.optimize_for_tracking()

print('O beam:')
print(line.particle_ref.show())

# Update beam parameters
beamParams.Nb = beamParams.Nb_O  # update to new oxygen intensity
print(beamParams)

# Then, check that space charge is installed correctly
sps_fma = fma_ions.FMA(output_folder=output_folder)


lprofile = xf.LongitudinalProfileQGaussian(
                number_of_particles = beamParams.Nb,
                sigma_z = beamParams.sigma_z,
                z0=0.,
                q_parameter=1.0)

# Install frozen space charge as base 
xf.install_spacecharge_frozen(line = line,
                    particle_ref = line.particle_ref,
                    longitudinal_profile = lprofile,
                    nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn,
                    sigma_z = beamParams.sigma_z,
                    num_spacecharge_interactions = 1080)

line.build_tracker(_context = context)

# Generate particles
particles = xp.generate_matched_gaussian_bunch(_context=context,
                num_particles=num_part, 
                total_intensity_particles=beamParams.Nb,
                nemitt_x=beamParams.exn, 
                nemitt_y=beamParams.eyn, 
                sigma_z= beamParams.sigma_z,
                particle_ref=line.particle_ref, 
                line=line)

x, y = sps_fma.track_particles(particles, line, which_context=which_context, save_tbt_data=False)
Qx, Qy, d = sps_fma.run_FMA(x, y, which_context=which_context)


# Tunes from Twiss
Qh_set = twiss_sps['qx']
Qv_set = twiss_sps['qy']

# Add interger tunes to fractional tunes 
Qx += int(twiss_sps['qx'])
Qy += int(twiss_sps['qy'])
        
# Make tune footprint, need plot range
plot_range  = [[26.0, 26.35], [26.0, 26.35]]

sps_fma.plot_FMA(d, Qx, Qy, Qh_set, Qv_set,'SPS', plot_range)

# Test default tracking with space charge on CPU context - then test plotting
#sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=100, num_turns=100, turn_print_interval=10)
#tbt = sps.track_SPS(ion_type='O', which_context='cpu', add_aperture=True, apply_kinetic_IBS_kicks=True)
#sps.plot_tracking_data(tbt, show_plot=True)
