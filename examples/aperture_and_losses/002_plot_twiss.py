"""
Example script to plot twiss functions
"""
import fma_ions
import matplotlib.pyplot as plt
import numpy as np

# Load sequence 
sps_seq = fma_ions.SPS_sequence_maker()
line, tw = sps_seq.load_xsuite_line_and_twiss()



fig1 = plt.figure(1, figsize=(7.4, 4.8), constrained_layout=True)
spbet = plt.subplot(2,1,1)
spdisp  = plt.subplot(2,1,2, sharex=spbet)

spbet.plot(tw.s, tw.betx, label='$\beta_{x}$')
spbet.plot(tw.s, tw.bety, label='$\beta_{y}$')
spbet.set_ylabel('$\beta_{x,y}$ [m]')
spbet.legend(fontsize=12)

spdisp.plot(tw.s, tw.dx, label='$D_{x}$')
spdisp.plot(tw.s, tw.dy, label='$D_{y}$')
spdisp.set_ylabel(r'$D_{x,y}$ [m]')
spdisp.set_xlabel('s [m]')
spdisp.legend(fontsize=12)

#fig1.suptitle(
#    r'$q_x$ = ' f'{tw.qx:.5f}' r' $q_y$ = ' f'{tw.qy:.5f}' '\n'
#    r"$Q'_x$ = " f'{tw.dqx:.2f}' r" $Q'_y$ = " f'{tw.dqy:.2f}'
#    r' $\gamma_{tr}$ = '  f'{1/np.sqrt(tw.momentum_compaction_factor):.2f}'
#)
#fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
spbet.set_xlim(-0.1, 1650)
fig1.savefig('SPS_Q26_twiss_function.png', dpi=350)
plt.show()
