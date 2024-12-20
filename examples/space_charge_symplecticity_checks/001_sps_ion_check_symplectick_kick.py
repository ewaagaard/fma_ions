"""
Example to generate five SPS Pb particles, spaced out in initial longitudinal zeta amplitude, with modified synchrotron tune for each particle

Ensure that longitudinal
"""
import numpy as np
import time
import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import fma_ions


def change_synchrotron_tune_by_factor(A, line, sigma_z, Nb):
    """
    Scale synchrotron tune Qs while keeping bucket half-height delta constant, also adjusting
    bunch length and bunch intensity accordingly for identical bunch filling factor and space charge effects

    Parameters
    ----------
    line : xtrack.Line
        line used in tracking
    A : float
        factor by which to scale the synchrotron tune
    sigma_z : float
        original bunch length
    Nb : float
        original bunch intensity

    Returns
    -------
    line_new : xtrack.Line
        line with updated RF voltage and harmonic
    sigma_z_new : float
        updated new bunch length
    Nb_new : float
        updated new bunch intensity
    """

    # Find RF cavity number 
    nn = 'actcse.31632' # 'actcse.31637' for protons
    
    line[nn].voltage *= A # scale voltage by desired factor
    line[nn].frequency *= A # in reality scale harmonic number, but translates directly to frequency
    sigma_z_new = sigma_z / A  # adjust bunch length such that space charge effects remain the same
    Nb_new = Nb / A # adjust bunch intensity such that space charge effects remain the same
    
    return line, sigma_z_new, Nb_new


# Initialize chosen context, number of turns, particles and space charge interactions
context = xo.ContextCpu(omp_num_threads='auto')
num_turns = 10_000
number_of_particles = 5
num_spacecharge_interactions = 1080
scale_factor_Qs = 2.0  # by how many times to scale the nominal synchrotron tune

# Beam parameters, will be used for space charge
Nb = 2.46e8 # bunch_intensity measured 2.46e8 Pb ions per bunch on 2023-10-16
sigma_z =  0.225
nemitt_x = 1.3e-6
nemitt_y = 0.9e-6

# Import line
sps_seq = fma_ions.SPS_sequence_maker()
line, _ = sps_seq.load_xsuite_line_and_twiss()
line, sigma_z, Nb = change_synchrotron_tune_by_factor(scale_factor_Qs, line, sigma_z, Nb) # update synchrotron tune, scale bucket length and SC parameters
twiss = line.twiss()
print('\nUpdated parameters to sigma_z = {:.4f} and Nb = {:.3e}'.format(sigma_z, Nb))
print('New Qs = {:.6f} when Qs changed by factor {}\n'.format(twiss['qs'], scale_factor_Qs))

# Generate particles spread out in lognitudinal space make linear spacing between close to center of RF bucket and to separatrix
zetas = np.linspace(0.05, 0.7 / scale_factor_Qs, num=number_of_particles)
p0 = xp.build_particles(line = line, particle_ref = line.particle_ref,
                            x_norm=0.1, y_norm=0.1, delta=0.0, zeta=zetas,
                            nemitt_x = nemitt_x, nemitt_y = nemitt_y, _context=context) # default transverse amplitude is 0.1 sigmas


# Install frozen space charge, emulating a Gaussian bunch
lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles = Nb,
        sigma_z = sigma_z,
        z0=0.,
        q_parameter=1.0)

# Install frozen space charge as base 
xf.install_spacecharge_frozen(line = line,
                   particle_ref = line.particle_ref,
                   longitudinal_profile = lprofile,
                   nemitt_x = nemitt_x, nemitt_y = nemitt_y,
                   sigma_z = sigma_z,
                   num_spacecharge_interactions = num_spacecharge_interactions)

line.build_tracker(_context = context)
line.enable_time_dependent_vars = True
line.track(p0.copy(), num_turns=num_turns, with_progress=True,
           log=xt.Log(zeta=lambda l, p: p.zeta.copy()))
log_no_kick = line.log_last_track

# The following lines are equivalent to setting configure_longitudinal_sc_kick=True
# in xf.install_spacecharge_frozen()
tt = line.get_table()
tt_sc = tt.rows[tt.element_type=='SpaceChargeBiGaussian']
for nn in tt_sc.name:
    line[nn].z_kick_num_integ_per_sigma = 5

line.track(p0.copy(), num_turns=num_turns, with_progress=True,
              log=xt.Log(zeta=lambda l, p: p.zeta.copy()))
log_with_kick = line.log_last_track

zeta_no_kick = np.stack(log_no_kick['zeta'])
zeta_with_kick = np.stack(log_with_kick['zeta'])


fig, ax = plt.subplots(1,1,figsize=(8,6), constrained_layout=True)
i_part_plot = -1
ax.plot(zeta_no_kick[:, i_part_plot], label='No z kick')
ax.plot(zeta_with_kick[:, i_part_plot], label='With z kick')
z0 = p0.zeta[i_part_plot]
ax.set_ylim([z0*0.99, z0*1.05])
ax.set_xlabel('Turns')
ax.set_ylabel('$\\zeta$ [m]')
ax.grid(alpha=0.5)
ax.legend()
fig.savefig('example_plots/SPS_Pb_check_symplectic_SC_kick.png', dpi=250)
plt.show()
