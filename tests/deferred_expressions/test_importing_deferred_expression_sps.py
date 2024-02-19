"""
Test loading the SPS lattice with deferred expressions to see which errors are encountered
"""
import fma_ions
import xtrack as xt
from cpymad.madx import Madx

line_to_test = 'sps'

# First load standard SPS sequence from https://gitlab.cern.ch/acc-models/acc-models-sps, or PS
madx = Madx()
if line_to_test=='sps':
    
    madx.call('SPS_LS2_2020-05-26.seq')
    madx.input('Beam, particle=ion, mass=193.7, charge=82, energy = 1415.72;')
    #           DPP:=BEAM->SIGE*(BEAM->ENERGY/BEAM->PC)^2;')  # --> this is the 
    madx.command.use(sequence='sps')
    
    line = xt.Line.from_madx_sequence(madx.sequence['sps'], deferred_expressions=True)
elif line_to_test=='ps':
    madx.call('PS_STANDARD_PR_YETS 2022-2023_20-MAR-2023.seq')
    madx.input('BEAM, PARTICLE=Pb54, MASS=0.931494 * 207.976/208., CHARGE=54./208., ENERGY=0.931494 * 207.976/208. + 0.072;')
    madx.command.use(sequence='ps')
    
    line = xt.Line.from_madx_sequence(madx.sequence['ps'], deferred_expressions=True)


# Test first loading line with deferred expressions
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_SPS_line_with_deferred_madx_expressions()

# Try making a line from the madx instance
madx = sps.load_simple_madx_seq(add_non_linear_magnet_errors=False)
line = xt.Line.from_madx_sequence(madx.sequence['sps'], deferred_expressions=True)

# Then try loading file with errors
line2, twiss2 = sps.load_SPS_line_with_deferred_madx_expressions(add_non_linear_magnet_errors=True)

