"""
Test loading the SPS lattice with deferred expressions to see which errors are encountered
"""
import fma_ions
import xtrack as xt

# Test first loading line with deferred expressions
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_SPS_line_with_deferred_madx_expressions()

# Try making a line from the madx instance
madx = sps.load_simple_madx_seq(add_non_linear_magnet_errors=False)
line = xt.Line.from_madx_sequence(madx.sequence['sps'], deferred_expressions=True)

# Then try loading file with errors
line2, twiss2 = sps.load_SPS_line_with_deferred_madx_expressions(add_non_linear_magnet_errors=True)

#madx = sps.load_simple_madx_seq(add_non_linear_magnet_errors=True)
#line = xt.Line.from_madx_sequence(madx.sequence['sps'], deferred_expressions=True)

