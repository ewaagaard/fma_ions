"""
Tester script to check SPS aperture and anomalous behaviour
"""
import numpy as np
import fma_ions
import xtrack as xt

aperture_min = 0.035

def print_smallest_aperture(line: xt.Line):
    """function to return and print smallest aperture values"""

    # Get aperture table
    a = line.check_aperture()
    a = a[a['is_aperture']] # remove elements without aperture

    # Loop over all elements to find aperture values
    x_ap = []
    y_ap = []
    s_ap = [] # location of aperture
    ind = []

    for i, ele in enumerate(a.element):
        if ele is not None:
            x_ap.append(ele.max_x)
            y_ap.append(ele.max_y)
            s_ap.append(a.s.iloc[i])
            ind.append(i)

    # Convert to numpy arrays
    x_ap = np.array(x_ap)
    y_ap = np.array(y_ap)
    s_ap = np.array(s_ap)
    ind = np.array(ind)

    # Find minimum aperture
    print('Minimum X aperture is x_min={} m at s={} m'.format(x_ap[np.argmin(x_ap)], s_ap[np.argmin(x_ap)]))
    print('Minimum Y aperture is y_min={} m at s={} m'.format(y_ap[np.argmin(y_ap)], s_ap[np.argmin(y_ap)]))

    return x_ap, y_ap, a

# Import SPS line
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()

# Generate aperture table
x_ap, y_ap, a = print_smallest_aperture(line)

# Sort and show how many values that are small
xx = np.sort(x_ap)
ii = x_ap[x_ap < aperture_min]
a = a.iloc[:-1] # drop last None row
aa = a.iloc[x_ap < aperture_min]
print(aa)

# Remove all apertures that are too small
mask = [True] * len(line.elements)
for i in aa.index:
    mask[i] = False
line = line.filter_elements(mask) # remove these elements

# Check aperture table of new line 
x_ap2, y_ap2, b = print_smallest_aperture(line)
