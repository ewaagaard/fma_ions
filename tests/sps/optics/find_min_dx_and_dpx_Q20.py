"""
Check optics functions of normal Q20 sequence, find point with minimal D'x and Dx
"""
import fma_ions
import matplotlib.pyplot as plt
import numpy as np

# Q20 optics
sps = fma_ions.SPS_sequence_maker(qx0=20.30, qy0=20.25, proton_optics='q20')
line, twiss = sps.load_xsuite_line_and_twiss()

print('Q20 (normal): Qx = {:.3f}, Qy = {:.3f}'.format(twiss['qx'], twiss['qy']))
print('Q20 start values: betx = {:3f} m, bety = {:3f} m, Dx = {:3f} m, Dxprime = {:.3f}m'.format(twiss.betx[0], twiss.bety[0], twiss.dx[0], twiss.dpx[0]))
print('Q20 max values: betx = {:3f} m, bety = {:3f} m, Dx = {:3f} m\n'.format(max(twiss.betx), max(twiss.bety), max(twiss.dx)))

# Q20 optics - cycled to minimum dpx
line2 = line.cycle(index_first_element=np.argmin(np.abs(twiss.dpx)))
twiss2 = line2.twiss()

print('Q20 (min dpx): Qx = {:.3f}, Qy = {:.3f}'.format(twiss2['qx'], twiss2['qy']))
print('Q20 cycled start values: betx = {:3f} m, bety = {:3f} m, Dx = {:3f} m, Dxprime = {:.3f}m'.format(twiss2.betx[0], twiss2.bety[0], twiss2.dx[0], twiss2.dpx[0]))
print('Q20 cycled max values: betx = {:3f} m, bety = {:3f} m, Dx = {:3f} m\n'.format(max(twiss2.betx), max(twiss2.bety), max(twiss2.dx)))

# Q20 optics - cycled to minimum dx
line3 = line.cycle(index_first_element=np.argmin(np.abs(twiss.dx)))
twiss3 = line3.twiss()

print('Q20 (min dx): Qx = {:.3f}, Qy = {:.3f}'.format(twiss3['qx'], twiss3['qy']))
print('Q20 cycled start values: betx = {:3f} m, bety = {:3f} m, Dx = {:3f} m, Dxprime = {:.3f}m'.format(twiss3.betx[0], twiss3.bety[0], twiss3.dx[0], twiss3.dpx[0]))
print('Q20 cycled max values: betx = {:3f} m, bety = {:3f} m, Dx = {:3f} m\n'.format(max(twiss3.betx), max(twiss3.bety), max(twiss3.dx)))


# Q20 optics - cycled to minimum dx AND dpx simultaneously
penalty = twiss.dx**2 + twiss.dpx**2
line4 = line.cycle(index_first_element=np.argmin(penalty))
twiss4 = line4.twiss()

print('Q20 (min dx and dpx): Qx = {:.3f}, Qy = {:.3f}'.format(twiss4['qx'], twiss4['qy']))
print('Q20 cycled start values: betx = {:3f} m, bety = {:3f} m, Dx = {:3f} m, Dxprime = {:.3f}m'.format(twiss4.betx[0], twiss4.bety[0], twiss4.dx[0], twiss4.dpx[0]))
print('Q20 cycled max values: betx = {:3f} m, bety = {:3f} m, Dx = {:3f} m\n'.format(max(twiss4.betx), max(twiss4.bety), max(twiss4.dx)))

# Plot Q20 optics - minimum dispersion
fig1, (spbet, spdisp, spdisp1) = plt.subplots(3, 1, figsize=(6.4, 4.8*1.5), sharex=True)

spbet.plot(twiss3.s, twiss3.betx)
spbet.plot(twiss3.s, twiss3.bety)
spbet.set_ylabel(r'$\beta_{x,y}$ [m]')

spdisp.plot(twiss3.s, twiss3.dx)
spdisp.plot(twiss3.s, twiss3.dy)
spdisp.set_ylabel(r'$D_{x,y}$ [m]')
spdisp.set_xlabel('s [m]')

spdisp1.plot(twiss3.s, twiss3.dpx)
spdisp1.plot(twiss3.s, twiss3.dpy)
spdisp1.set_ylabel(r"$D_{x,y}$' [m]")
spdisp1.set_xlabel('s [m]')

fig1.suptitle(
    r'Q20 min dx: $q_x$ = ' f'{twiss3.qx:.4f}' r' $q_y$ = ' f'{twiss3.qy:.4f}' '\n'
    r"$Q'_x$ = " f'{twiss3.dqx:.2f}' r" $Q'_y$ = " f'{twiss3.dqy:.2f}'
    r' $\gamma_{tr}$ = '  f'{1/np.sqrt(twiss3.momentum_compaction_factor):.2f}'
)
fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
fig1.tight_layout()

# Plot Q20 optics - cycled
fig2, (spbet2, spdisp2, spdisp22) = plt.subplots(3, 1, figsize=(6.4, 4.8*1.5), sharex=True)

spbet2.plot(twiss4.s, twiss4.betx)
spbet2.plot(twiss4.s, twiss4.bety)
spbet2.set_ylabel(r'$\beta_{x,y}$ [m]')


spdisp2.plot(twiss4.s, twiss4.dx)
spdisp2.plot(twiss4.s, twiss4.dy)
spdisp2.set_ylabel(r'$D_{x,y}$ [m]')
spdisp2.set_xlabel('s [m]')

spdisp22.plot(twiss4.s, twiss4.dpx)
spdisp22.plot(twiss4.s, twiss4.dpy)
spdisp22.set_ylabel(r"$D_{x,y}$' [m]")
spdisp22.set_xlabel('s [m]')

fig2.suptitle(
    r'Q20 min(dx, dpx): $q_x$ = ' f'{twiss4.qx:.4f}' r' $q_y$ = ' f'{twiss4.qy:.4f}' '\n'
    r"$Q'_x$ = " f'{twiss4.dqx:.2f}' r" $Q'_y$ = " f'{twiss4.dqy:.2f}'
    r' $\gamma_{tr}$ = '  f'{1/np.sqrt(twiss4.momentum_compaction_factor):.2f}'
)
fig2.subplots_adjust(left=.15, right=.92, hspace=.27)
fig2.tight_layout()

plt.show()