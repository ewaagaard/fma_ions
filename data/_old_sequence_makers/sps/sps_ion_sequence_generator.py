"""
SPS WITH PB IONS FLAT BOTTOM - matching chromaticity with sextupoles with correct weight

by Elias Waagaard 
"""
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

from cpymad.madx import Madx
import json

optics = '/home/elwaagaa/cernbox/PhD/Projects/acc-models-sps'

#### Initiate MADX sequence and call the sequence and optics file ####
madx = Madx()
madx.call("{}/sps.seq".format(optics))
madx.call("{}/strengths/lhc_ion.str".format(optics))
madx.call("{}/beams/beam_lhc_ion_injection.madx".format(optics))
madx.use(sequence='sps')

# SPS PB ION CHROMA VALUES: not displayed on acc-model, extracted from PTC Twiss 
qx0 = 26.30
qy0 = 26.25
dq1 = -3.460734474533172e-09 
dq2 = -3.14426538905229e-09

# Flatten line
madx.use("sps")
madx.input("seqedit, sequence=SPS;")
madx.input("flatten;")
madx.input("endedit;")
madx.use("sps")
madx.input("select, flag=makethin, slice=5, thick=false;")
madx.input("makethin, sequence=sps, style=teapot, makedipedge=True;")

# Use correct tune and chromaticity matching macros
madx.call("{}/toolkit/macro.madx".format(optics))
madx.use('sps')
madx.exec(f"sps_match_tunes({qx0},{qy0});")
madx.exec("sps_define_sext_knobs();")
madx.exec("sps_set_chroma_weights_q26();")
madx.input(f"""match;
global, dq1={dq1};
global, dq2={dq2};
vary, name=qph_setvalue;
vary, name=qpv_setvalue;
jacobian, calls=10, tolerance=1e-25;
endmatch;""")

# Create Xsuite line
madx.use(sequence='sps')
twiss_thin = madx.twiss()  

line = xt.Line.from_madx_sequence(madx.sequence['sps'])
line.build_tracker()
madx_beam = madx.sequence['sps'].beam

particle_sample = xp.Particles(
        p0c = madx_beam.pc*1e9,
        q0 = madx_beam.charge,
        mass0 = madx_beam.mass*1e9)

line.particle_ref = particle_sample

#### SET CAVITY VOLTAGE - with info from Hannes
# 6x200 MHz cavities: actcse, actcsf, actcsh, actcsi (3 modules), actcsg, actcsj (4 modules)
# acl 800 MHz cavities
# acfca crab cavities
# Ions: all 200 MHz cavities: 1.7 MV, h=4653
harmonic_nb = 4653
nn = 'actcse.31632'

# MADX sequence 
madx.sequence.sps.elements[nn].lag = 0
madx.sequence.sps.elements[nn].volt = 3.0*particle_sample.q0 # different convention between madx and xsuite
madx.sequence.sps.elements[nn].freq = madx.sequence['sps'].beam.freq0*harmonic_nb

# Xsuite sequence 
line[nn].lag = 0  # 0 if below transition
line[nn].voltage =  3.0e6 # In Xsuite for ions, do not multiply by charge as in MADX
line[nn].frequency = madx.sequence['sps'].beam.freq0*1e6*harmonic_nb

# Save MADX sequence
madx.command.save(sequence='sps', file='SPS_2021_Pb_ions_matched_with_RF.seq', beam=True)

# Save Xsuite sequence
with open('SPS_2021_Pb_ions_matched_with_RF.json', 'w') as fid:
    json.dump(line.to_dict(), fid, cls=xo.JEncoder)