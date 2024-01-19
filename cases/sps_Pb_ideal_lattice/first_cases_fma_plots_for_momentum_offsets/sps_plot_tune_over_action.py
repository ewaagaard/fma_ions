"""
SPS Pb ions: plot tune over action from generated turn-by-turn data
"""
import fma_ions

# On-momentum case

fma_sps = fma_ions.FMA(output_folder='output_Pb_on_momentum_ideal_lattice')

# Load Twiss and plot normalized phase space
sps_seq = fma_ions.SPS_sequence_maker()
sps_line, twiss_sps = sps_seq.load_xsuite_line_and_twiss(Qy_frac=25)

Jx, Jy, Qx, Qy = fma_sps.plot_tune_over_action(twiss_sps, also_show_plot=False, case_name='z0 = 0')


# Off-momentum case

fma_sps2 = fma_ions.FMA(output_folder='output_Pb_off_momentum_ideal_lattice_z0_0dot1')

Jx2, Jy2, Qx2, Qy2 = fma_sps2.plot_tune_over_action(twiss_sps, also_show_plot=True, case_name='z0 = 0.1')