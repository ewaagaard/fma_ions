"""
Check optics functions of normal Q20 sequence, find point with minimal and maximal Dx --> the D'x is kept as fixed as possible
"""
import fma_ions
import matplotlib.pyplot as plt
import numpy as np

# Q20 optics
sps = fma_ions.SPS_sequence_maker(qx0=20.30, qy0=20.25, proton_optics='q20')
line, twiss = sps.load_xsuite_line_and_twiss()

print('Q20 (normal): Qx = {:.3f}, Qy = {:.3f}'.format(twiss['qx'], twiss['qy']))
print('Q20 start values: Dx = {:3f} m, Dxprime = {:.3f}m\n'.format(twiss.dx[0], twiss.dpx[0]))

# Q20 optics - cycled to minimum dpx 
penalty2 = np.abs(twiss.dx)
line2 = line.cycle(index_first_element=np.argmin(penalty2))
twiss2 = line2.twiss()
print('Q20 (min dx): Qx = {:.3f}, Qy = {:.3f}'.format(twiss2['qx'], twiss2['qy']))
print('Q20 cycled start values: Dx = {:3f} m, Dxprime = {:.3f}m\n'.format(twiss2.dx[0], twiss2.dpx[0]))

# Q20 optics - cycled to maximum dpx
penalty3 = -twiss.dx + (twiss.dpx - twiss2.dpx[0])**2
line3 = line.cycle(index_first_element=np.argmin(penalty3))
twiss3 = line3.twiss()

print('Q20 (max dpx): Qx = {:.3f}, Qy = {:.3f}'.format(twiss3['qx'], twiss3['qy']))
print('Q20 cycled start values: Dx = {:3f} m, Dxprime = {:.3f}m\n'.format(twiss3.dx[0], twiss3.dpx[0]))


# Plot Q20 optics - minimum dpx
fig1, (spbet, spdisp, spdisp1) = plt.subplots(3, 1, figsize=(6.4, 4.8*1.5), sharex=True)

spbet.plot(twiss2.s, twiss2.betx)
spbet.plot(twiss2.s, twiss2.bety)
spbet.set_ylabel(r'$\beta_{x,y}$ [m]')

spdisp.plot(twiss2.s, twiss2.dx)
spdisp.plot(twiss2.s, twiss2.dy)
spdisp.set_ylabel(r'$D_{x,y}$ [m]')
spdisp.set_xlabel('s [m]')

spdisp1.plot(twiss2.s, twiss2.dpx)
spdisp1.plot(twiss2.s, twiss2.dpy)
spdisp1.set_ylabel(r"$D_{x,y}$' [m]")
spdisp1.set_xlabel('s [m]')

fig1.suptitle(
    r'Q20 min dx: $q_x$ = ' f'{twiss2.qx:.4f}' r' $q_y$ = ' f'{twiss2.qy:.4f}' '\n'
    r"$Q'_x$ = " f'{twiss2.dqx:.2f}' r" $Q'_y$ = " f'{twiss2.dqy:.2f}'
    r' $\gamma_{tr}$ = '  f'{1/np.sqrt(twiss2.momentum_compaction_factor):.2f}'
)
fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
fig1.tight_layout()

# Plot Q20 optics - maximu dpx
fig2, (spbet2, spdisp2, spdisp22) = plt.subplots(3, 1, figsize=(6.4, 4.8*1.5), sharex=True)

spbet2.plot(twiss3.s, twiss3.betx)
spbet2.plot(twiss3.s, twiss3.bety)
spbet2.set_ylabel(r'$\beta_{x,y}$ [m]')


spdisp2.plot(twiss3.s, twiss3.dx)
spdisp2.plot(twiss3.s, twiss3.dy)
spdisp2.set_ylabel(r'$D_{x,y}$ [m]')
spdisp2.set_xlabel('s [m]')

spdisp22.plot(twiss3.s, twiss3.dpx)
spdisp22.plot(twiss3.s, twiss3.dpy)
spdisp22.set_ylabel(r"$D_{x,y}$' [m]")
spdisp22.set_xlabel('s [m]')

fig2.suptitle(
    r'Q20 max dx: $q_x$ = ' f'{twiss3.qx:.4f}' r' $q_y$ = ' f'{twiss3.qy:.4f}' '\n'
    r"$Q'_x$ = " f'{twiss3.dqx:.2f}' r" $Q'_y$ = " f'{twiss3.dqy:.2f}'
    r' $\gamma_{tr}$ = '  f'{1/np.sqrt(twiss3.momentum_compaction_factor):.2f}'
)
fig2.subplots_adjust(left=.15, right=.92, hspace=.27)
fig2.tight_layout()

plt.show()