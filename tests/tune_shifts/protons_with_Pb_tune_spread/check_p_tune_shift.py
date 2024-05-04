"""
Check tune shift p beam in Q26 for SPS, find emittances to get tune shift similar to Pb beam
Use sequence generate from
https://gitlab.cern.ch/elwaagaa/xsuite-sps-ps-sequence-benchmarker/-/blob/master/SPS_sequence/SPS_2021_sequence_generator_Protons.py
"""
import fma_ions
import injector_model
import numpy as np
import json
import xtrack as xt
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "figure.titlesize": 20,
    }
)
# Predicted tune shift for Pb beams - tested from 'check_Pb_tune_shift.py'
dQx_Pb = -0.15788
dQy_Pb = -0.21436

# First, check that reference particle is correct
with open('SPS_2021_Protons_matched_with_RF.json', 'r') as fid:
    loaded_dct = json.load(fid)
line = xt.Line.from_dict(loaded_dct)
twiss = line.twiss()

print('p beam:')
print(line.particle_ref.show())

# Update beam parameters to proton parameters
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = 1e11 
beamParams.exn = 2.5e-6 # starting value
beamParams.eyn = 2.5e-6
beamParams.sigma_z = 0.22
beamParams.delta = 1e-3
print(beamParams)

# Then calculate it manually
sc = injector_model.SC_Tune_Shifts()
gamma = line.particle_ref.gamma0[0]
beta = sc.beta(gamma)
r0 = line.particle_ref.get_classical_particle_radius0()
print('Classical particle radius p: {}'.format(r0))

# Create function to get tune shiifts - assume round beams
def get_tune_shifts_from_emittance(norm_emitt):
    
    # Update emittances
    beamParams.exn = norm_emitt
    beamParams.eyn = norm_emitt

    twiss_xtrack_interpolated, sigma_x, sigma_y = sc.interpolate_Twiss_table(twissTableXsuite=twiss, 
                                                                                particle_ref=line.particle_ref,
                                                                                line_length=line.get_length(), beamParams=beamParams)

    # Space charge perveance
    K_sc = (2 * r0 * beamParams.Nb) / (np.sqrt(2*np.pi) * beamParams.sigma_z * beta**2 * gamma**3)

    # Numerically integrate SC lattice integral
    integrand_x = twiss_xtrack_interpolated['betx'] / (sigma_x * (sigma_x + sigma_y))  
    integrand_y = twiss_xtrack_interpolated['bety'] / (sigma_y * (sigma_x + sigma_y)) 

    dQx0 = - K_sc / (4 * np.pi) * np.trapz(integrand_x, x = twiss_xtrack_interpolated['s'])
    dQy0 = - K_sc / (4 * np.pi) * np.trapz(integrand_y, x = twiss_xtrack_interpolated['s'])
    print('SC tune shift: dQx = {:.5f}, dQy = {:.5f}'.format(dQx0, dQy0))

    return dQx0, dQy0

# Initiate vectors and iterate function over values
emitt_array = np.linspace(0.1e-6, 2e-6, num=25)
dQx = np.zeros(len(emitt_array))
dQy = np.zeros(len(emitt_array))

for i, emitt in enumerate(emitt_array):
    dQx[i], dQy[i] = get_tune_shifts_from_emittance(emitt)

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.plot(emitt_array, dQx, label=r'$dQ_{{x}}$')
ax.plot(emitt_array, dQy, label=r'$dQ_{{y}}$')
ax.axhline(y=dQx_Pb, ls='--', c='g', label=r'Pb $dQ_{{x}}$')
ax.axhline(y=dQy_Pb, ls='--', c='violet', label=r'Pb $dQ_{{y}}$')
ax.set_ylabel('Tune shift')
ax.set_xlabel(r'$\varepsilon^{{n}}_{{x,y}}$ [m rad]')
ax.legend()
fig.savefig('p_tune_shifts_over_emittances_vs_Pb.png', dpi=250)
plt.show()