"""
Initial script to generate a SPS Q20 proton line
"""
import fma_ions
import xobjects as xo
import json

# Generate line without magnet errors
sps = fma_ions.SPS_sequence_maker(qx0=20.13, qy0=20.18, ion_type='proton', proton_optics='q20')
line = sps.generate_xsuite_seq()

with open('sps_q20_proton_line.json', 'w') as fid:
   json.dump(line.to_dict(), fid, cls=xo.JEncoder)

# Generate line with magnet errors
sps2 = fma_ions.SPS_sequence_maker(qx0=20.13, qy0=20.18, ion_type='proton', proton_optics='q20')
line2 = sps2.generate_xsuite_seq(add_non_linear_magnet_errors=True)

with open('sps_q20_proton_line_with_errors.json', 'w') as fid:
   json.dump(line2.to_dict(), fid, cls=xo.JEncoder)