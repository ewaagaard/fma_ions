"""
Check tune shift p beam in Q26 for SPS, find emittances to get tune shift similar to O beam
Construct objective function, then use minimizer

"""
import fma_ions
import injector_model
import numpy as np
import json
import xtrack as xt
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
# Predicted tune shift for O beams - tested from 'check_O_tune_shift.py'
dQx_O = -0.19109 
dQy_O = -0.27610

# First, check that reference particle is correct
with open('../SPS_2021_Protons_matched_with_RF.json', 'r') as fid:
    loaded_dct = json.load(fid)
line = xt.Line.from_dict(loaded_dct)
twiss = line.twiss()

print('p beam:')
print(line.particle_ref.show())

# Update beam parameters to proton parameters
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = 1e11 
en0 = [1.0e-6, 1.0e-6] # starting guess for emittances value
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
def get_tune_shifts_from_emittance(en):
    """
    Function to return tune shifts from a given set of emittances

    Parameters
    ----------
    en : list
        array with normalized emittances exn, eyn

    Returns
    -------
    dQx, dQy : float
        tune shift in each plane
    """
    
    # Update emittances
    beamParams.exn = en[0]
    beamParams.eyn = en[1]

    twiss_xtrack_interpolated, sigma_x, sigma_y = sc.interpolate_Twiss_table(twissTableXsuite=twiss, 
                                                                                particle_ref=line.particle_ref,
                                                                                line_length=line.get_length(), beamParams=beamParams)
    # Space charge perveance
    K_sc = (2 * r0 * beamParams.Nb) / (np.sqrt(2*np.pi) * beamParams.sigma_z * beta**2 * gamma**3)

    # Numerically integrate SC lattice integral
    integrand_x = twiss_xtrack_interpolated['betx'] / (sigma_x * (sigma_x + sigma_y))  
    integrand_y = twiss_xtrack_interpolated['bety'] / (sigma_y * (sigma_x + sigma_y)) 

    dQx = - K_sc / (4 * np.pi) * np.trapz(integrand_x, x = twiss_xtrack_interpolated['s'])
    dQy = - K_sc / (4 * np.pi) * np.trapz(integrand_y, x = twiss_xtrack_interpolated['s'])
    print('SC tune shift: dQx = {:.5f}, dQy = {:.5f} for exn = {:.2e}, eyn = {:.2e}'.format(dQx, dQy, en[0], en[1]))

    return dQx, dQy


def objective_func(en):

    # Find tune shifts from emittances
    dQx, dQy = get_tune_shifts_from_emittance(en)
    
    # Define objective function to minimize square of difference to desired tune shift
    cost = (dQx - dQx_O)**2 + (dQy - dQy_O)**2    
    
    return cost

# Minimize cost function to find emittances corresponding to correct tune shifts
result = minimize(objective_func, en0,
                  method='nelder-mead', tol=1e-5, options={'maxiter':100})

print('\nFinal result: exn = {:.3e}, eyn = {:.3e}'.format(result.x[0], result.x[1]))
en_final = [result.x[0], result.x[1]]
dQx_f, dQy_f = get_tune_shifts_from_emittance(en_final)
print('with tune shifts dQx, dQy = {:4f}, {:.4f}'.format(dQx_f, dQy_f))