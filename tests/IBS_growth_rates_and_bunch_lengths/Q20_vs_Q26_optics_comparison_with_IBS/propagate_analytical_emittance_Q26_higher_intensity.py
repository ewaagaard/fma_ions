"""
Propagate analytical emittance for Q26 optics with parameters similar to nominal case
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt
import xpart as xp
import pickle
from xibs.inputs import BeamParameters, OpticsParameters
from xibs.analytical import NagaitsevIBS

get_first_parameters_from_particles = False

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 20,
        "axes.titlesize": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 14.5,
        "ytick.labelsize": 14.5,
        "legend.fontsize": 12.5,
        "figure.titlesize": 20,
    }
)

# Number of turns and particles
num_turns = 100_000
num_part = 50_000
ibs_step = 5000  # frequency at which to re-compute the growth rates in [turns]
beamParams = fma_ions.BeamParameters_SPS_Binomial_2016()  # Binomial parameters
beamParams.Nb *= 10 # increas intensity

# Create sequence generators for Q26
sps_q26 = fma_ions.SPS_sequence_maker(qx0=26.3, qy0=26.25, proton_optics='q26')
line, twiss = sps_q26.load_xsuite_line_and_twiss()
    
# Generate particles for line
particles = xp.generate_matched_gaussian_bunch(
                num_particles=num_part, 
                total_intensity_particles=beamParams.Nb,
                nemitt_x=beamParams.exn, 
                nemitt_y=beamParams.eyn, 
                sigma_z= beamParams.sigma_z,
                particle_ref=line.particle_ref, 
                line=line)


######### IBS kinetic kicks and analytical model #########
beamparams = BeamParameters.from_line(line, n_part=beamParams.Nb)
opticsparams = OpticsParameters.from_line(line) # read from line without space  charge
NIBS = NagaitsevIBS(beamparams, opticsparams)

# Initialize the dataclass
analytical_tbt = fma_ions.Records_Growth_Rates.init_zeroes(num_turns)
twiss = line.twiss()
analytical_tbt.update_at_turn(0, particles, twiss)


for turn in range(1, num_turns):
    
    # ----- Potentially re-compute the IBS growth rates and kick coefficients ----- #
    if (turn % ibs_step == 0) or (turn == 1):
        print(f"Turn {turn:d}: re-computing diffusion and friction terms")

        # Compute analytical values from those at the previous turn
        growth_rates = NIBS.growth_rates(
            analytical_tbt.epsilon_x[turn - 1],
            analytical_tbt.epsilon_y[turn - 1],
            analytical_tbt.sigma_delta[turn - 1],
            analytical_tbt.bunch_length[turn - 1],
        )

    # ----- Compute analytical Emittances from previous turn values & update records----- #
    ana_emit_x, ana_emit_y, ana_sig_delta, ana_bunch_length = NIBS.emittance_evolution(
        analytical_tbt.epsilon_x[turn - 1],
        analytical_tbt.epsilon_y[turn - 1],
        analytical_tbt.sigma_delta[turn - 1],
        analytical_tbt.bunch_length[turn - 1],
    )
    analytical_tbt.epsilon_x[turn] = ana_emit_x
    analytical_tbt.epsilon_y[turn] = ana_emit_y
    analytical_tbt.sigma_delta[turn] = ana_sig_delta
    analytical_tbt.bunch_length[turn] = ana_bunch_length
    

# Convert turns to seconds
turns = np.arange(num_turns, dtype=int)  # array of turns
turns_per_sec = 1 / twiss.T_rev0
ctime = turns / turns_per_sec

# Convert to dictionary
tbt_dict = analytical_tbt.to_dict()
tbt_dict['exn'] = tbt_dict['ex'] * twiss.beta0 * twiss.gamma0
tbt_dict['eyn'] = tbt_dict['ey'] * twiss.beta0 * twiss.gamma0
tbt_dict['ctime'] = ctime

with open('analytical_Q26.pickle', 'wb') as handle:
    pickle.dump(tbt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Plot the results
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (9, 4))

# Plot from analytical values - convert to normalized emittance
ax1.plot(ctime, analytical_tbt.epsilon_x * 1e6 * particles.beta0[0] * particles.gamma0[0], ls='-', 
            lw=2.5, label=r'$\varepsilon_{{x}}^{{n}}$: {}'.format('Q26'))
ax1.plot(ctime, analytical_tbt.epsilon_y * 1e6 * particles.beta0[0] * particles.gamma0[0], ls='--',
            lw=2.5,  label=r'$\varepsilon_{{y}}^{{n}}$: {}'.format('Q26'))
ax2.plot(ctime, analytical_tbt.bunch_length, lw=2.5, label=r'$\sigma_{{z}}$: {}'.format('Q26'))

for ax in (ax1, ax2):
    ax.set_xlabel('Time [s]')

ax1.set_ylabel(r'$\varepsilon_{x, y}^{n}$ [$\mu$m]')
ax2.set_ylabel(r"Bunch length [m]")
ax1.legend(fontsize=12.5, loc='center left')
ax2.legend(fontsize=12.5, loc='upper right')

f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()