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

# SPS PB ION CHROMA VALUES: not displayed on acc-model, extract from PTC Twiss
# - for SPS protons, dq is slightly positive, but for Pb ions they seem to be matched to zero 
madx.input('''        
dp = 0;
order = 2;
ptc_create_universe;
ptc_create_layout, time=false, model=2, exact=true, method=6, nst=3;
select, flag=ptc_twiss, clear;
select, flag=ptc_twiss, column=name,keyword,s,x,px,beta11,alfa11,beta22,alfa22,disp1,disp2,mu1,mu2,energy,l,angle,K1L,K2L,K3L,HKICK,SLOT_ID;    

use, sequence=SPS;
ptc_twiss, closed_orbit, icase=56, no=order, deltap=dp, table=ptc_twiss, summary_table=ptc_twiss_summary, normal;
ptc_end;
           ''')
ptc_table_thick = madx.table.ptc_twiss_summary
ptc_twiss_thick = madx.ptc_twiss

qx0 = 26.30
qy0 = 26.25

dq1 = madx.table.ptc_twiss_summary['dq1'][0]
dq2 = madx.table.ptc_twiss_summary['dq2'][0]

madx.use(sequence='sps')
twiss_thick = madx.twiss()

print("PTC: thick    " f"Qx  = {ptc_table_thick['q1'][0]:.8f}"                     f"   Qy = {ptc_table_thick['q2'][0]:.8f}")
print("MAD-X thick:  " f"Qx  = {twiss_thick.summary['q1']:.8f}"           f"   Qy = {twiss_thick.summary['q2']:.8f}")

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

madx.input('''        
dp = 0;
order = 2;
ptc_create_universe;
ptc_create_layout, time=false, model=2, exact=true, method=6, nst=3;
select, flag=ptc_twiss, clear;
select, flag=ptc_twiss, column=name,keyword,s,x,px,beta11,alfa11,beta22,alfa22,disp1,disp2,mu1,mu2,energy,l,angle,K1L,K2L,K3L,HKICK,SLOT_ID;    

use, sequence=SPS;
ptc_twiss, closed_orbit, icase=56, no=order, deltap=dp, table=ptc_twiss, summary_table=ptc_twiss_summary, normal;
ptc_end;
           ''')

ptc_table_thin = madx.table.ptc_twiss_summary
ptc_twiss_thin = madx.ptc_twiss

# Twiss command of thin sequence 
madx.use(sequence='sps')
twiss_thin = madx.twiss()    

# Compare thin and thick sequence
beta0 = madx.sequence['sps'].beam.beta
print("SPS PB ions: MADX thin vs thick sequence:")
print("PTC: thick    " f"Qx  = {ptc_table_thick['q1'][0]:.8f}"                     f"   Qy = {ptc_table_thick['q2'][0]:.8f}")
print("MAD-X thick:  " f"Qx  = {twiss_thick.summary['q1']:.8f}"           f"   Qy = {twiss_thick.summary['q2']:.8f}")
print("PTC: thin     " f"Qx  = {ptc_table_thin['q1'][0]:.8f}"                     f"   Qy = {ptc_table_thin['q2'][0]:.8f}")
print("MAD-X thin:   " f"Qx  = {twiss_thin.summary['q1']:.8f}"            f"   Qy = {twiss_thin.summary['q2']:.8f}")
print("\nPTC thick:    " f"Q'x = {ptc_table_thick['dq1'][0]:.8f}"                     f"  Q'y = {ptc_table_thick['dq2'][0]:.8f}")
print("MAD-X thick:  " f"Q'x = {twiss_thick.summary['dq1']*beta0:.7f}"    f"   Q'y = {twiss_thick.summary['dq2']*beta0:.7f}")
print("PTC thin:     " f"Q'x = {ptc_table_thin['dq1'][0]:.8f}"                     f"  Q'y = {ptc_table_thin['dq2'][0]:.8f}")
print("MAD-X thin:   " f"Q'x = {twiss_thin.summary['dq1']*beta0:.7f}"     f"   Q'y = {twiss_thin.summary['dq2']*beta0:.7f}")
print("\nPTC thick:    " f"alpha_p = {ptc_table_thick['alpha_c'][0]:.8f}")
print("MAD-X thick:  " f"alpha_p = {twiss_thick.summary.alfa:.7f}")
print("PTC thin:    " f"alpha_p = {ptc_table_thin['alpha_c'][0]:.8f}")
print("MAD-X thin:   " f"alpha_p = {twiss_thin.summary.alfa:.7f}")



### XSUITE TRACKER AND TWISS 

# Perform Twiss command with MADX
madx.use(sequence='sps')
line = xt.Line.from_madx_sequence(madx.sequence['sps'])
madx_beam = madx.sequence['sps'].beam

particle_sample = xp.Particles(
        p0c = madx_beam.pc*1e9,
        q0 = madx_beam.charge,
        mass0 = madx_beam.mass*1e9)

line.particle_ref = particle_sample

### Perform Twiss command from tracker and save Xtrack sequence in json format
line.build_tracker()
twiss_xtrack = line.twiss(method='4d')  


print("\nPB IONS: XTRACK vs MADX sequence:")
print("PTC thin:    " f"Qx  = {ptc_table_thin['q1'][0]:.8f}"                     f"   Qy = {ptc_table_thin['q2'][0]:.8f}")
print("MAD-X thin:   " f"Qx  = {twiss_thin.summary['q1']:.8f}"            f"   Qy  = {twiss_thin.summary['q2']:.8f}")
print("Xsuite:       " f"Qx  = {twiss_xtrack['qx']:.8f}"                  f"   Qy  = {twiss_xtrack['qy']:.8f}\n")
print("\nPTC thin:    " f"Q'x = {ptc_table_thin['dq1'][0]:.8f}"                     f"  Q'y = {ptc_table_thin['dq2'][0]:.8f}")
print("MAD-X thin:   " f"Q'x = {twiss_thin.summary['dq1']*beta0:.7f}"     f"   Q'y = {twiss_thin.summary['dq2']*beta0:.7f}")
print("Xsuite:       " f"Q'x = {twiss_xtrack['dqx']:.7f}"                 f"   Q'y = {twiss_xtrack['dqy']:.7f}\n")
print("\nPTC thin:    " f"alpha_p = {ptc_table_thin['alpha_c'][0]:.8f}")
print("MAD-X thin:   " f"alpha_p = {twiss_thin.summary.alfa:.7f}")
print("Xsuite:       " f"alpha_p = {twiss_xtrack['momentum_compaction_factor']:.7f}")

# Save MADX without RF
# madx.command.save(sequence='sps', file='../SPS_sequence/SPS_2021_Pb_ions_thin_matched.seq', beam=True)

### SET CAVITY VOLTAGE - with info from Hannes
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

# Now rematch chromaticities again after RF has been introduced 

# Xsuite sequence 
line[nn].lag = 0  # 0 if below transition
line[nn].voltage =  3.0e6 # In Xsuite for ions, do not multiply by charge as in MADX
line[nn].frequency = madx.sequence['sps'].beam.freq0*1e6*harmonic_nb

# Rematch Xsuite line for the same purpose - but which are the knobs? 
# ... no variables can be found in tracker.vars
# first need to solve deferred expression = True from MADX->Xsuite conversion 
"""
tracker.match(
    vary=[
        xt.Vary('kLSDA.b2', step=1e-8),
        xt.Vary('kLSFA.b2', step=1e-8),
    ],
    targets = [
        xt.Target('dqx', madx.globals['qpx'], tol=1e-5),
        xt.Target('dqy', madx.globals['qpy'], tol=1e-5)])
"""

### Check Twiss commands that they still make sense
madx.input('''        
dp = 0;
order = 2;
ptc_create_universe;
ptc_create_layout, time=false, model=2, exact=true, method=6, nst=3;
select, flag=ptc_twiss, clear;
select, flag=ptc_twiss, column=name,keyword,s,x,px,beta11,alfa11,beta22,alfa22,disp1,disp2,mu1,mu2,energy,l,angle,K1L,K2L,K3L,HKICK,SLOT_ID;    

use, sequence=SPS;
ptc_twiss, closed_orbit, icase=56, no=order, deltap=dp, table=ptc_twiss, summary_table=ptc_twiss_summary, normal;
ptc_end;
           ''')

ptc_table_thin_RF = madx.table.ptc_twiss_summary
ptc_twiss_thin_RF = madx.ptc_twiss

madx.use(sequence = 'sps')
twiss_thin_RF = madx.twiss()

twiss_xtrack_RF = line.twiss()  

print("\nPB IONS WITH RF: XTRACK vs MADX sequence:")
print("PTC thin:     "  f"Qx  = {ptc_table_thin_RF['q1'][0]:.8f}"                     f"   Qy = {ptc_table_thin_RF['q2'][0]:.8f}")
print("MAD-X thin:   " f"Qx  = {twiss_thin_RF.summary['q1']:.8f}"            f"   Qy  = {twiss_thin_RF.summary['q2']:.8f}")
print("Xsuite:       " f"Qx  = {twiss_xtrack_RF['qx']:.8f}"                  f"   Qy  = {twiss_xtrack_RF['qy']:.8f}\n")
print("\nPTC thin:    " f"Q'x = {ptc_table_thin_RF['dq1'][0]:.8f}"                     f"  Q'y = {ptc_table_thin_RF['dq2'][0]:.8f}")
print("MAD-X thin:   " f"Q'x = {twiss_thin_RF.summary['dq1']*beta0:.7f}"     f"   Q'y = {twiss_thin_RF.summary['dq2']*beta0:.7f}")
print("Xsuite:       " f"Q'x = {twiss_xtrack_RF['dqx']:.7f}"                 f"   Q'y = {twiss_xtrack_RF['dqy']:.7f}\n")
print("\nPTC thin:    " f"alpha_p = {ptc_table_thin_RF['alpha_c'][0]:.8f}")
print("MAD-X thin:   " f"alpha_p = {twiss_thin_RF.summary.alfa:.7f}")
print("Xsuite:       " f"alpha_p = {twiss_xtrack_RF['momentum_compaction_factor']:.7f}")

### SAVE SEQUENCES 

# Save MADX sequence
# madx.command.save(sequence='sps', file='../SPS_sequence/SPS_2021_Pb_ions_matched_with_RF.seq', beam=True)

# Save Xsuite sequence
with open('SPS_2021_Pb_ions_matched_with_RF.json', 'w') as fid:
    json.dump(line.to_dict(), fid, cls=xo.JEncoder)

