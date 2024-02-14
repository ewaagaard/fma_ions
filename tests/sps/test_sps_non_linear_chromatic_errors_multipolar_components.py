"""
Test differences between SPS lines with and without non-linear magnet errors - check magnet components in both cases
"""
import fma_ions
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
import xpart as xp

# Load lines with and without magnet errors
sps = fma_ions.SPS_sequence_maker()
'''
madx = sps.load_simple_madx_seq(add_non_linear_magnet_errors=False)
# Make line without magnet errors and add reference particle
line = xt.Line.from_madx_sequence(madx.sequence['sps'], apply_madx_errors=False)
m_in_eV, p_inj_SPS = sps.generate_SPS_beam()
line.particle_ref = xp.Particles(
        p0c = p_inj_SPS,
        q0 = sps.Q_SPS,
        mass0 = m_in_eV)
'''
line = sps.generate_xsuite_seq(add_non_linear_magnet_errors=False)
twiss = line.twiss()

# With magnet errors 
sps2 = fma_ions.SPS_sequence_maker()

'''
madx2 = sps2.load_simple_madx_seq(add_non_linear_magnet_errors=True)
# Make line with magnet errors and add reference particle
line2 = xt.Line.from_madx_sequence(madx2.sequence['sps'], apply_madx_errors=True)
line2.particle_ref = xp.Particles(
        p0c = p_inj_SPS,
        q0 = sps.Q_SPS,
        mass0 = m_in_eV)
'''
line2 = sps2.generate_xsuite_seq(add_non_linear_magnet_errors=True)
twiss2 = line2.twiss()

my_dict = line.to_dict()
my_dict2 = line2.to_dict()
d =  my_dict["elements"]
d2 =  my_dict2["elements"]

# Check if these are different
for key, value in d.items():
    if value['__class__'] == 'Multipole':
        if not np.array_equal(value['knl'][-1], d2[key]['knl'][-1]):
            print('{}: ideal: {} vs error: {}'.format(key, value['knl'][-1], d2[key]['knl'][-1]))
