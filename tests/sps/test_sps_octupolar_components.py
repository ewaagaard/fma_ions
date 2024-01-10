"""
Check octupolar k3 components from Twiss with SPS Pb sequence used for FMA 
"""
import fma_ions

# Initialize FMA object
fma_ps = fma_ions.FMA()

# Test Twiss and tune adjustments of SPS 
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()
print('\nSPS beam: new tunes Qx = {:.4f}, Qy = {:.4f}\n'.format(twiss['qx'], twiss['qy']))
 
my_dict = line.to_dict()
d =  my_dict["elements"]

# Print all octupolar components present
for key, value in d.items():
    if value['__class__'] == 'Multipole' and value['_order'] >= 3:
        print('{}: knl = {}'.format(key, value['knl']))



