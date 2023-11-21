import numpy as np
from cpymad.madx import Madx
import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata
from PyNAFF import naff
import NAFFlib
from statisticalEmittance.statisticalEmittance import statisticalEmittance 

num_turns = 1200

####################
# Choose a context #
####################

context = xo.ContextCpu()
# context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

print(context)

mad = Madx()
mad.call('psb_flat_bottom.madx')

line= xt.Line.from_madx_sequence(mad.sequence['psb1'])
line.particle_ref=xp.Particles(mass0=xp.PROTON_MASS_EV,
                               gamma0=mad.sequence.psb1.beam.gamma)
particle_ref = line.particle_ref

nemitt_x=1.5e-6
nemitt_y=1e-6
bunch_intensity=50e10
sigma_z=16.9


########## FROM THIS PART DOWN, SIMILAR TO SPS FMA EXAMPLE ##########
num_spacecharge_interactions = 540
tol_spacecharge_position = 1e-2
mode = 'frozen'


# Install spacecharge interactions (frozen) #
lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles=bunch_intensity,
        sigma_z=sigma_z,
        z0=0.,
        q_parameter=1.)


xf.install_spacecharge_frozen(line=line,
                   particle_ref=line.particle_ref,
                   longitudinal_profile=lprofile,
                   nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                   sigma_z=sigma_z,
                   num_spacecharge_interactions=num_spacecharge_interactions,
                   tol_spacecharge_position=tol_spacecharge_position)

                   
# Build tracker
line.build_tracker(_context=context)
line_sc_off = line.filter_elements(exclude_types_starting_with='SpaceCh')
twiss_xtrack = line_sc_off.twiss()  # optics adapted for sequence w/o SC
twiss_xtrack_with_sc = line.twiss()

x_norm, y_norm, _, _ = xp.generate_2D_polar_grid(
    theta_range=(0.01, np.pi/2-0.01),
    ntheta = 20,
    r_range = (0.1, 7),
    nr = 30)

particles = xp.build_particles(line=line, particle_ref=line.particle_ref,
                               x_norm=x_norm, y_norm=y_norm, delta=0,
                               nemitt_x=nemitt_x, nemitt_y=nemitt_y)
n_part = len(particles.x)

# Initialize dictionaries for particles turn-by-turn tracking
x_tbt = np.zeros((n_part , num_turns), dtype=np.float64)
y_tbt = np.zeros((n_part , num_turns), dtype=np.float64)
px_tbt = np.zeros((n_part , num_turns), dtype=np.float64)
py_tbt = np.zeros((n_part , num_turns), dtype=np.float64)

# Also for the non-dispersive components
x_tbt_ndc = np.zeros((n_part , num_turns), dtype=np.float64)
y_tbt_ndc = np.zeros((n_part , num_turns), dtype=np.float64)
px_tbt_ndc = np.zeros((n_part , num_turns), dtype=np.float64)
py_tbt_ndc = np.zeros((n_part , num_turns), dtype=np.float64)

# Define object to calculate statistical emittance 
em = statisticalEmittance(particles)

Qx_ndc = np.zeros(N_footprint)
Qy_ndc = np.zeros(N_footprint)


# Track the particles
for ii in range(num_turns):
    print('Tracking turn {}'.format(ii))

    em.setInputDistribution(particles)

    # Calculate the coordinates and beam matrix excluding dispersive components
    em.betatronicMatrices()
    x_tbt_ndc[:, ii] = em.coordinateMatrixBetatronic[0]
    px_tbt_ndc[:, ii] = em.coordinateMatrixBetatronic[1]
    y_tbt_ndc[:, ii] = em.coordinateMatrixBetatronic[2]
    py_tbt_ndc[:, ii] = em.coordinateMatrixBetatronic[3]

    line.track(particles)

# Remove dispersive component - remove dispersion*delta from each particle
x_tbt_noDP = x_tbt - twiss_xtrack_with_sc['dx'][0]*(x_tbt.transpose() - particles.delta).transpose()
px_tbt_noDP = px_tbt - twiss_xtrack_with_sc['dpx'][0]*(px_tbt.transpose() - particles.delta).transpose()
y_tbt_noDP = y_tbt - twiss_xtrack_with_sc['dy'][0]*(y_tbt.transpose() - particles.delta).transpose()
py_tbt_noDP = py_tbt - twiss_xtrack_with_sc['dpy'][0]*(py_tbt.transpose() - particles.delta).transpose()


xn_tbt = x_tbt_noDP/np.sqrt(twiss_xtrack_with_sc['betx'][0])
pxn_tbt = -twiss_xtrack_with_sc['alfx'][0]/np.sqrt(twiss_xtrack_with_sc['betx'][0])*x_tbt_noDP \
          + np.sqrt(twiss_xtrack_with_sc['betx'][0])*px_tbt_noDP
yn_tbt = y_tbt_noDP/np.sqrt(twiss_xtrack_with_sc['bety'][0])
pyn_tbt = -twiss_xtrack_with_sc['alfy'][0]/np.sqrt(twiss_xtrack_with_sc['bety'][0])*y_tbt_noDP \
          + np.sqrt(twiss_xtrack_with_sc['bety'][0])*py_tbt_noDP

# Set threshold for synchrotron radiation frequency to filter out
Qmin = 0.05

# Add the estimates tunes - also remember to subtract mean 
for i_part in range(N_footprint):

    # Pick out dominant tune from dispersion-corrected betatronic matrix data, also comparing with complex x_tbt px_tbt pair
    Qx_ndc_tune = NAFFlib.get_tunes(x_tbt_ndc[i_part, :] - np.mean(x_tbt_ndc[i_part, :]), 2)[0]  
    Qx_ndc[i_part] = Qx_ndc_tune[np.argmax(Qx_ndc_tune>Qmin)]
    
    Qy_ndc_tune = NAFFlib.get_tunes(y_tbt_ndc[i_part, :] - np.mean(y_tbt_ndc[i_part, :]), 2)[0]  
    Qy_ndc[i_part] = Qy_ndc_tune[np.argmax(Qy_ndc_tune>Qmin)]


fig23 = plt.figure(figsize=(10,7))
fig23.suptitle('Method 3: Tune footprint: betatronic matrices \n D-corrected (green) and D-corrected complex (blue)', fontsize=16)
ax23 = fig23.add_subplot(1,1,1)
ax23.plot(Qx_ndc+6.0, Qy_ndc+6.0, 'go', markersize=4.5, alpha=0.2)
ax23.plot(Qx_ndc_complex+6.0, Qy_ndc_complex+6.0, 'bo', markersize=1.5, alpha=0.2)
ax23.set_ylabel('$Q_{y}$')
ax23.set_xlabel('$Q_{x}$')

"""
# Track the particles
i = 0
for turn in range(num_turns):
    print('Tracking turn {}'.format(i))
    i += 1

    # Track the particles
    line.track(particles)

    # Add X and Y data
    tbt_data['x'].append(particles.x)
    tbt_data['y'].append(particles.y)


# FMA 
Qx1, Qx = [], []
Qy1, Qy = [], []

# X and Y position in matrix format, and centroids over n_turns 

x = np.array(tbt_data['x'])
x_bar = np.mean(x, axis=1)
y = np.array(tbt_data['y'])
y_bar = np.mean(y, axis=1)

# Extract all tunes stepping through all particles - n_turns rows x n_part columns
for i in range(n_part):
    try:
        Qx1.append(naff(x[:, i] - np.mean(x[:, i]), turns=600)[0][1])
    except:
        Qx1.append(np.nan)
    try:
        Qy1.append(naff(y[:, i] - y_bar, turns=600)[0][1])
    except:
        Qy1.append(np.nan)
    try:
        Qx.append(naff(x[:, i] - x_bar, skipTurns=599, turns=600)[0][1])
    except:
        Qx.append(np.nan)
    try:
        Qy.append(naff(y[:, i] - y_bar, skipTurns=599, turns=600)[0][1])
    except:
        Qy.append(np.nan)
Qx1=4.0+np.array(Qx1)
Qx=4.0+np.array(Qx)
Qy1=4.0+np.array(Qy1)
Qy=4.0+np.array(Qy)
d = np.log(np.sqrt( (Qx-Qx1)**2 + (Qy-Qy1)**2 ))

fig=plt.figure(figsize=(9, 6))
plt.title('Tune Diagram', fontsize='20')
plt.scatter(Qx, Qy,4, d, 'o',lw = 0.1,zorder=10, cmap=plt.cm.jet)
plt.plot([4.2],[4.4],'ko',zorder=1e5)
plt.xlabel('$\mathrm{Q_x}$', fontsize='20')
plt.ylabel('$\mathrm{Q_y}$', fontsize='20')
plt.tick_params(axis='both', labelsize='18')
plt.clim(-20.5,-4.5)
cbar=plt.colorbar()
cbar.set_label('d',fontsize='18')
cbar.ax.tick_params(labelsize='18')
plt.tight_layout()
#plt.savefig('test_fma.png')

fig2=plt.figure()
XX,YY = np.meshgrid(np.unique(x[:,0]), np.unique(y[:,0]))
Z = griddata((x[:,0],y[:,0]), d, (XX,YY), method='linear')
Zm = np.ma.masked_invalid(Z)
fig2.suptitle('Initial Distribution', fontsize='20')
plt.pcolormesh(XX,YY,Zm,cmap=plt.cm.jet)
# plt.scatter(x[:,0],y[:,0],4, d, 'o',lw = 0.1,zorder=10, cmap=plt.cm.jet)
plt.tick_params(axis='both', labelsize='18')
plt.xlabel('x [m]', fontsize='20')
plt.ylabel('y [m]', fontsize='20')
plt.clim(-20.5,-4.5)
cbar=plt.colorbar()
cbar.set_label('d',fontsize='18')
cbar.ax.tick_params(labelsize='18')
# fig2.savefig('test_initial_distribution.png')
"""