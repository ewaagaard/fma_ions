"""
Tester script to generate SPS sequence with aperture, and check this aperture
"""
import fma_ions
import acc_lib
import matplotlib.pyplot as plt


# Generate SPS sequence maker, make sure aperture exists 
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss(add_aperture=True)
df = line.check_aperture()

# From MADX instance, plot aperture with updated emittances
madx = sps.load_simple_madx_seq(add_aperture=True)
madx.sequence.sps.exn = 1.3e-6
madx.sequence.sps.eyn = 0.9e-6

#Activate the aperture for the Twiss flag to include it in Twiss command! 
madx.use(sequence='sps')
madx.input('select,flag=twiss,clear;')
madx.select(flag='twiss', column=['name','s','l',
              'lrad','angle','k1l','k2l','k3l','k1sl','k2sl','k3sl','hkick','vkick','kick','tilt',
              'betx','bety','alfx','alfy','dx','dpx','dy','dpy','mux','muy','x','y','px','py','t','pt',
              'wx','wy','phix','phiy','n1','ddx','ddy','ddpx','ddpy',
              'keyword','aper_1','aper_2','aper_3','aper_4',
              'apoff_1','apoff_2',
              'aptol_1','aptol_2','aptol_3','apertype','mech_sep'])

twiss = madx.twiss()
new_pos_x, aper_neat_x = acc_lib.madx_tools.get_apertures_real(twiss)
new_pos_y, aper_neat_y = acc_lib.madx_tools.get_apertures_real(twiss, axis='vertical')

# Print know aperture element for SPS, make sure that it is a LimitRectEllipse
print(df.iloc[61447])

#### Plot the beam envelope and aperture
fig = plt.figure(figsize=(10,7))
ax = acc_lib.madx_tools.plot_envelope(fig, madx, twiss)
acc_lib.madx_tools.plot_apertures_real(ax, new_pos_x, aper_neat_x)
fig.suptitle('SPS Pb ions - horizontal aperture', fontsize=22)
plt.show()

