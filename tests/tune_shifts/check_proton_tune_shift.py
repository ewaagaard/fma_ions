"""
Check tune shift of SPS proton beams with correct input parameters
"""
import fma_ions
import injector_model
import numpy as np

# First, check that reference particle is correct
sps = fma_ions.SPS_sequence_maker(ion_type='proton') 
line, twiss_sps = sps.load_xsuite_line_and_twiss()

print('Proton beam:')
print(line.particle_ref.show())

# Update beam parameters
beamParams = fma_ions.BeamParameters_SPS_Proton()
beamParams.Nb = 1.0e11 
beamParams.exn = 0.5e-6
beamParams.eyn = 0.5e-6
beamParams.sigma_z = 0.22 #m
beamParams.delta = 1e-3


# Then calculate it manually
sc = injector_model.SC_Tune_Shifts()
gamma = line.particle_ref.gamma0[0]
beta = sc.beta(gamma)
r0 = line.particle_ref.get_classical_particle_radius0()

# Calculated interpolated twiss table
twiss_xtrack_interpolated, sigma_x, sigma_y = sc.interpolate_Twiss_table(twissTableXsuite=twiss_sps, 
                                                                            particle_ref=line.particle_ref,
                                                                            line_length=line.get_length(), beamParams=beamParams)

# Space charge perveance
K_sc = (2 * r0 * beamParams.Nb) / (np.sqrt(2*np.pi) * beamParams.sigma_z * beta**2 * gamma**3)

# Numerically integrate SC lattice integral
integrand_x = twiss_xtrack_interpolated['betx'] / (sigma_x * (sigma_x + sigma_y))  
integrand_y = twiss_xtrack_interpolated['bety'] / (sigma_y * (sigma_x + sigma_y)) 

dQx0 = - K_sc / (4 * np.pi) * np.trapz(integrand_x, x = twiss_xtrack_interpolated['s'])
dQy0 = - K_sc / (4 * np.pi) * np.trapz(integrand_y, x = twiss_xtrack_interpolated['s'])

print('\nWith: sigma_z = {:.3f} m, Nb = {:.3e}, exn = {:.2e}, eyn = {:.2e}:'.format(beamParams.sigma_z, beamParams.Nb, beamParams.exn, beamParams.eyn))
print('\nSC tune shift protons: dQx = {:.5f}, dQy = {:.5f}'.format(dQx0, dQy0))