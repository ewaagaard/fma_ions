"""
Plot optics functions of Q26 vs Q20, check if plots look reasonable
"""
import fma_ions
import matplotlib.pyplot as plt
import numpy as np

# Q26 optics
sps = fma_ions.SPS_sequence_maker(qx0=26.30, qy0=26.25, proton_optics='q26')
line, twiss = sps.load_xsuite_line_and_twiss()

# Q20 optics
sps2 = fma_ions.SPS_sequence_maker(qx0=20.30, qy0=20.25, proton_optics='q20')
line2, twiss2 = sps2.load_xsuite_line_and_twiss()

print('Q26: Qx = {:.3f}, Qy = {:.3f}'.format(twiss['qx'], twiss['qy']))
print('Q20: Qx = {:.3f}, Qy = {:.3f}'.format(twiss2['qx'], twiss2['qy']))

print('\nQ26 max values: betx = {:3f} m, bety = {:3f} m, Dx = {:3f} m'.format(max(twiss.betx), max(twiss.bety), max(twiss.dx)))
print('Q20 max values: betx = {:3f} m, bety = {:3f} m, Dx = {:3f} m'.format(max(twiss2.betx), max(twiss2.bety), max(twiss2.dx)))

# Plot Q26 optics
fig1, (spbet, spdisp, spdisp1) = plt.subplots(3, 1, figsize=(6.4, 4.8*1.5), sharex=True)

spbet.plot(twiss.s, twiss.betx)
spbet.plot(twiss.s, twiss.bety)
spbet.set_ylabel(r'$\beta_{x,y}$ [m]')

#spco.plot(twiss.s, twiss.x)
#spco.plot(twiss.s, twiss.y)
#spco.set_ylabel(r'(Closed orbit)$_{x,y}$ [m]')

spdisp.plot(twiss.s, twiss.dx)
spdisp.plot(twiss.s, twiss.dy)
spdisp.set_ylabel(r'$D_{x,y}$ [m]')
spdisp.set_xlabel('s [m]')

spdisp1.plot(twiss.s, twiss.dpx)
spdisp1.plot(twiss.s, twiss.dpy)
spdisp1.set_ylabel(r"$D_{x,y}$' [m]")
spdisp1.set_xlabel('s [m]')

fig1.suptitle(
    r'Q26: $q_x$ = ' f'{twiss.qx:.4f}' r' $q_y$ = ' f'{twiss.qy:.4f}' '\n'
    r"$Q'_x$ = " f'{twiss.dqx:.2f}' r" $Q'_y$ = " f'{twiss.dqy:.2f}'
    r' $\gamma_{tr}$ = '  f'{1/np.sqrt(twiss.momentum_compaction_factor):.2f}'
)
fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
fig1.tight_layout()

# Plot Q20 optics
fig2, (spbet2, spdisp2, spdisp22) = plt.subplots(3, 1, figsize=(6.4, 4.8*1.5), sharex=True)

spbet2.plot(twiss2.s, twiss2.betx)
spbet2.plot(twiss2.s, twiss2.bety)
spbet2.set_ylabel(r'$\beta_{x,y}$ [m]')

#spco2.plot(twiss2.s, twiss2.x)
#spco2.plot(twiss2.s, twiss2.y)
#spco2.set_ylabel(r'(Closed orbit)$_{x,y}$ [m]')

spdisp2.plot(twiss2.s, twiss2.dx)
spdisp2.plot(twiss2.s, twiss2.dy)
spdisp2.set_ylabel(r'$D_{x,y}$ [m]')
spdisp2.set_xlabel('s [m]')

spdisp22.plot(twiss2.s, twiss2.dpx)
spdisp22.plot(twiss2.s, twiss2.dpy)
spdisp22.set_ylabel(r"$D_{x,y}$' [m]")
spdisp22.set_xlabel('s [m]')

fig2.suptitle(
    r'Q20: $q_x$ = ' f'{twiss2.qx:.4f}' r' $q_y$ = ' f'{twiss2.qy:.4f}' '\n'
    r"$Q'_x$ = " f'{twiss2.dqx:.2f}' r" $Q'_y$ = " f'{twiss2.dqy:.2f}'
    r' $\gamma_{tr}$ = '  f'{1/np.sqrt(twiss2.momentum_compaction_factor):.2f}'
)
fig2.subplots_adjust(left=.15, right=.92, hspace=.27)
fig2.tight_layout()

plt.show()