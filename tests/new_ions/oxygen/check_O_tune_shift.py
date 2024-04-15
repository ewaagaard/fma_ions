"""
Check tune shift of SPS O beams with correct input parameters
"""
import fma_ions
import injector_model
import numpy as np

# First, check that reference particle is correct
sps = fma_ions.SPS_sequence_maker(26.30, 26.19, ion_type='O', Q_PS=4., Q_SPS=8., m_ion=15.9949) 
line, twiss_sps = sps.load_xsuite_line_and_twiss()

print('O beam:')
print(line.particle_ref.show())

# Update beam parameters
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = 1e9 # beamParams.Nb_O  # update to new oxygen intensity
beamParams.delta = 1e-3
print(beamParams)

# Find tune shift from Injector Chain
injector_chain = injector_model.InjectorChain_v2()
df = injector_chain.calculate_LHC_bunch_intensity_all_ion_species()
injector_chain.init_ion('O')
Nb_spaceChargeLimitSPS, dQx, dQy = injector_chain.SPS_SC_limit(Nb_max=beamParams.Nb)
print('SC tune shift 1: dQx = {:.5f}, dQy = {:.5f}'.format(dQx, dQy))

# Then calculate it manually
sc = injector_model.SC_Tune_Shifts()
gamma = line.particle_ref.gamma0[0]
beta = sc.beta(gamma)
r0 = line.particle_ref.get_classical_particle_radius0()
print('Classical particle radius O: {}'.format(r0))

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
print('SC tune shift 2: dQx = {:.5f}, dQy = {:.5f}'.format(dQx0, dQy0))