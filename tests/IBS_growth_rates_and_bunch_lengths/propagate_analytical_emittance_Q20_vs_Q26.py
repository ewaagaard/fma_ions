"""
Propagate analytical emittance for Q20 and Q26 optics
- example following https://fsoubelet.github.io/xibs/gallery/demo_analytical_nagaitsev_emittances.html
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt
import xpart as xp
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
num_turns = 1_000_000
num_part = 1_000_000
ibs_step = 5000  # frequency at which to re-compute the growth rates in [turns]
beamParams = fma_ions.BeamParameters_SPS_Binomial_2016()  # Binomial parameters
beamParams.sigma_z = 0.225 # shorten bunch slightly, for Gaussian shape

# Create sequence generators for Q20 and Q26
sps_q20 = fma_ions.SPS_sequence_maker(qx0=20.3, qy0=20.25, proton_optics='q20')
sps_q26 = fma_ions.SPS_sequence_maker(qx0=26.3, qy0=26.25, proton_optics='q26')

line_q20, twiss_q20 = sps_q20.load_xsuite_line_and_twiss()
line_q26, twiss_q26 = sps_q26.load_xsuite_line_and_twiss()
lines = [line_q20, line_q26]
optics = ['Q20', 'Q26']
dfs = []

for i, line in enumerate(lines):
    
    print('\n\nOptics: {}\n'.format(optics[i]))
    
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
    if get_first_parameters_from_particles:
        analytical_tbt.update_at_turn(0, particles, twiss)
    else:
        analytical_tbt.epsilon_x[0] = 1.79e-07
        analytical_tbt.epsilon_y[0] = 1.24e-07
        analytical_tbt.sigma_delta[0] = 0.00046 # reasonable value for SPS
        analytical_tbt.bunch_length[0] = beamParams.sigma_z
        analytical_tbt.Nb[0] = beamParams.Nb 


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
    
    # Append dataclass
    dfs.append(analytical_tbt)
    
    
# Plot the results
turns = np.arange(num_turns, dtype=int)  # array of turns
fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(15, 7))

for i, analytical_tbt in enumerate(dfs):
    # Plot from analytical values - convert to normalized emittance
    axs["epsx"].plot(turns, analytical_tbt.epsilon_x * 1e6 * particles.beta0[0] * particles.gamma0[0], lw=2.5, label=optics[i])
    axs["epsy"].plot(turns, analytical_tbt.epsilon_y * 1e6 * particles.beta0[0] * particles.gamma0[0], lw=2.5, label=optics[i])
    axs["sigd"].plot(turns, analytical_tbt.sigma_delta * 1e3, lw=2.5, label=optics[i])
    axs["bl"].plot(turns, analytical_tbt.bunch_length, lw=2.5, label=optics[i])

# Axes parameters
axs["epsx"].set_ylabel(r"$\varepsilon_x^{n}$ [$\mu$m]")
axs["epsy"].set_ylabel(r"$\varepsilon_y^{n}$ [$\mu$m]")
axs["sigd"].set_ylabel(r"$\sigma_{\delta}$ [$10^{-3}$]")
axs["bl"].set_ylabel(r"Bunch length [m]")

for axis in (axs["epsy"], axs["bl"]):
    axis.yaxis.set_label_position("right")
    axis.yaxis.tick_right()

for axis in (axs["sigd"], axs["bl"]):
    axis.set_xlabel("Turn Number")

for axis in axs.values():
    axis.yaxis.set_major_locator(plt.MaxNLocator(3))
    #axis.legend(loc=9, ncols=4)
axs["epsy"].legend()
fig.align_ylabels((axs["epsx"], axs["sigd"]))
fig.align_ylabels((axs["epsy"], axs["bl"]))
plt.tight_layout()

# Emittances and bunch length - smaller plot
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (9, 4))

# Convert turns to seconds
turns_per_sec = 1 / twiss.T_rev0
ctime = turns / turns_per_sec

# Plot from analytical values - convert to normalized emittance
colors = ['royalblue', 'darkorange']

for i, analytical_tbt in enumerate(dfs):
    ax1.plot(ctime, analytical_tbt.epsilon_x * 1e6 * particles.beta0[0] * particles.gamma0[0], ls='-', color=colors[i], 
             lw=2.5, label=r'$\varepsilon_{{x}}^{{n}}$: {}'.format(optics[i]))
    ax1.plot(ctime, analytical_tbt.epsilon_y * 1e6 * particles.beta0[0] * particles.gamma0[0], ls='--', color=colors[i], 
             lw=2.5,  label=r'$\varepsilon_{{y}}^{{n}}$: {}'.format(optics[i]))
    ax2.plot(ctime, analytical_tbt.bunch_length, color=colors[i], lw=2.5, label=r'$\sigma_{{z}}$: {}'.format(optics[i]))

for ax in (ax1, ax2):
    ax.set_xlabel('Time [s]')

ax1.set_ylabel(r'$\varepsilon_{x, y}^{n}$ [$\mu$m]')
ax2.set_ylabel(r"Bunch length [m]")
ax1.legend(fontsize=12.5, loc='center left')
ax2.legend(fontsize=12.5, loc='upper right')

f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
f.savefig('Analytical_emittance_evolution_Q20_vs_Q26_IBS_step_100_turns.png', dpi=250)
plt.show()