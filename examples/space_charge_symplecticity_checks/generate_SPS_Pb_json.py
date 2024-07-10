"""
Script to generate SPS sequence json file for Pb ions, used for varying synchrotron tune
"""
import fma_ions
import json
import xobjects as xo

sps = fma_ions.SPS_sequence_maker(26.30, 26.19)
line = sps.generate_xsuite_seq(save_xsuite_seq=False)

with open('SPS_Pb_2021.json', 'w') as fid:
    json.dump(line.to_dict(), fid, cls=xo.JEncoder)