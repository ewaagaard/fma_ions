"""
Example script to plot twiss functions
"""
import fma_ions
import matplotlib.pyplot as plt
import numpy as np

# Generate sequence as used in simulations
sps_seq = fma_ions.SPS_sequence_maker()
line = sps_seq.generate_xsuite_seq()
tw = line.twiss()

# Generate figure
fig1 = plt.figure(1, figsize=(7.4, 4.8), constrained_layout=True)
spbet = plt.subplot(2,1,1)
spdisp  = plt.subplot(2,1,2, sharex=spbet)

spbet.plot(tw.s, tw.betx, label='$\\beta_{x}$')
spbet.plot(tw.s, tw.bety, label='$\\beta_{y}$')
spbet.set_ylabel(r'$\beta_{x,y}$ [m]')
spbet.legend(fontsize=12, loc='upper right')
spbet.grid(alpha=0.45)
plt.setp(spbet.get_xticklabels(), visible=False)

spdisp.plot(tw.s, tw.dx, label='$D_{x}$')
spdisp.plot(tw.s, tw.dy, label='$D_{y}$')
spdisp.set_ylabel('$D_{x,y}$ [m]')
spdisp.set_xlabel('s [m]')
spdisp.legend(fontsize=12, loc='upper right')
spdisp.grid(alpha=0.45)

spbet.set_xlim(-0.1, 1650)
fig1.savefig('SPS_Q26_twiss_function.png', dpi=350)
plt.show()
