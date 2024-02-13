"""
Test aperture attached to LEIR 
"""
import fma_ions
import acc_lib
import matplotlib.pyplot as plt

leir = fma_ions.LEIR_sequence_maker()
madx = leir.load_madx(make_thin=True, add_aperture=True)

madx.input('select,flag=twiss,clear;')
madx.input('SELECT, FLAG=TWISS, COLUMN=NAME,KEYWORD,S,L, BETX,ALFX,X,DX,PX,DPX,MUX,BETY,ALFY,Y,DY,PY,DPY,MUY,APER_1,APER_2,K1l,RE11,RE12,RE21,RE22,RE33,RE34,RE43,RE44,RE16,RE26;')

twiss = madx.twiss()

# Plot the envelope
fig = plt.figure(figsize=(10,7))
ax = acc_lib.madx_tools.plot_envelope(fig, madx, twiss, seq_name='leir', axis='horizontal')
aperture_position, aperture = acc_lib.madx_tools.get_apertures_real(twiss, axis='horizontal')
acc_lib.madx_tools.plot_apertures_real(ax, aperture_position, aperture) 

# See if anything has been plotted!
plt.show()