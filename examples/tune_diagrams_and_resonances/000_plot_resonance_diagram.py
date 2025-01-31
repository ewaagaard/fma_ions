"""
Plot simple resonance diagram of typical Q26 SPS Pb case 
"""
import matplotlib.pyplot as plt
import numpy as np
import fma_ions

# Define parameters
plot_range = [[25.95, 26.52], [25.95, 26.52]]
plot_order = 4
periodicity = 6

# Create resonance diagram object
fig = plt.figure(figsize=(7,6), constrained_layout=True)
tune_diagram = fma_ions.resonance_lines(plot_range[0],
            plot_range[1], np.arange(1, plot_order+1), periodicity)
tune_diagram.plot_resonance(figure_object = fig, interactive=False)
fig.savefig('000_SPS_resonance_diagram', dpi=250)
plt.show()