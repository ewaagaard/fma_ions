"""
Tester script to generate SPS sequence - test removing 
"""
import fma_ions

# First generate sequence for Qy = 26.25
sps = fma_ions.SPS_sequence_maker(qy0=26.25)
line, twiss = sps.load_xsuite_line_and_twiss(use_symmetric_lattice=True)

# Also test loading the MADX instance with the symmetric lattice and print remaining QFA / QDA
madx = sps.load_simple_madx_seq(use_symmetric_lattice=True)

dash = '-' * 65
header = '\n{:<27} {:>12} {:>15} {:>8}\n{}'.format("Element", "Location", "Type", "Length", dash)
print("\nRemaining QFAs or QDAs")
print(header)
for ele in madx.sequence['sps'].elements:
    if ele.name[:3] == 'qfa' or ele.name[:3] == 'qda':   
        print('{:<27} {:>12.6f} {:>15} {:>8.3}'.format(ele.name, ele.at, ele.base_type.name, ele.length))
print(dash)
