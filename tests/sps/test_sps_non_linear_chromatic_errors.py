"""
Test differences between SPS lines with and without non-linear magnet errors to reproduce chromaticity
"""
import fma_ions
import matplotlib.pyplot as plt
import numpy as np


# Load lines with and without magnet errors
sps = fma_ions.SPS_sequence_maker()
madx = sps.load_simple_madx_seq(add_non_linear_magnet_errors=False)
line, twiss = sps.load_xsuite_line_and_twiss(add_non_linear_magnet_errors=False)

# With magnet errors 
sps2 = fma_ions.SPS_sequence_maker()
madx2 = sps2.load_simple_madx_seq(add_non_linear_magnet_errors=True)
line2, twiss2 = sps2.load_xsuite_line_and_twiss(add_non_linear_magnet_errors=True)

### Compare non-linear chromatic behaviour of ring 
delta_values = np.arange(-0.006, 0.006, 0.001)
qx_values = np.zeros([4, len(delta_values)])
qy_values = np.zeros([4, len(delta_values)])

# Check MADX values
for i, delta in enumerate(delta_values):
    print(f"\nMADX Working on {i} of delta values {len(delta_values)}")

    # MADX thin sequence - for some reason madx.twiss() does not take deltap as input argument    
    madx.input(f'''
               use,sequence=sps;
               twiss, DELTAP={delta};
              ''')
    qx_values[0, i] = madx.table.summ['q1'][0]
    qy_values[0, i] = madx.table.summ['q2'][0]
    
    madx2.input(f'''
               use,sequence=sps;
               twiss, DELTAP={delta};
              ''')
    qx_values[1, i] = madx2.table.summ['q1'][0]
    qy_values[1, i] = madx2.table.summ['q2'][0]


# Check xtrack sequences 
for i, delta in enumerate(delta_values):
    print(f"\nXtrack Working on {i} of delta values {len(delta_values)}")
    # Xtrack
    print("Testing Xtrack twiss...")
    tt = line.twiss(method='4d', delta0=delta)
    qx_values[2, i] = tt.qx
    qy_values[2, i] = tt.qy
    
    tt_2 = line2.twiss(method='4d', delta0=delta)
    qx_values[3, i] = tt_2.qx
    qy_values[3, i] = tt_2.qy

# Plot the result
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,7))
fig.suptitle('SPS Pb ions: Non-linear chromatic tests')
ax1.plot(delta_values, qx_values[0, :], marker='o', c='r', label='MADX no errors')
ax1.plot(delta_values, qx_values[1, :], marker='v', c='darkred', alpha=0.7, label='MADX magnet errors')
ax1.plot(delta_values, qx_values[2, :], marker='o', c='b', label='Xtrack no errors')
ax1.plot(delta_values, qx_values[3, :], marker='v', c='darkblue', alpha=0.7, label='Xtrack magnet errors')
ax1.legend(fontsize=14)
ax1.set_xticklabels([])
ax1.set_ylabel('$Q_{x}$')
ax2.plot(delta_values, qy_values[0, :], marker='o', c='r', label='MADX no errors')
ax2.plot(delta_values, qy_values[1, :], marker='v', c='darkred', alpha=0.7, label='MADX magnet errors')
ax2.plot(delta_values, qy_values[2, :], marker='o', c='b', label='Xtrack no errors')
ax2.plot(delta_values, qy_values[3, :], marker='v', c='darkblue', alpha=0.7, label='Xtrack magnet errors')
ax2.set_ylabel('$Q_{y}$')
ax2.set_xlabel('$\delta$')
plt.show()