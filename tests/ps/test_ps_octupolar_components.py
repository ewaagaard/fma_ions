"""
Check octupolar k3 components from Twiss with PS Pb sequence used for FMA 
"""
import fma_ions

# Initialize FMA object
fma_ps = fma_ions.FMA()

# Test Twiss and tune adjustments of PS 
ps = fma_ions.PS_sequence_maker(qx0=6.19, qy0=6.14)
line = ps.generate_xsuite_seq()
twiss = line.twiss()
print('\nPS beam: new tunes Qx = {:.4f}, Qy = {:.4f}\n'.format(twiss['qx'], twiss['qy']))
 
my_dict = line.to_dict()
d =  my_dict["elements"]

# Print all octupolar components present
for key, value in d.items():
    if value['__class__'] == 'Multipole' and value['_order'] >= 3:
        print('{}: knl = {}'.format(key, value['knl']))



