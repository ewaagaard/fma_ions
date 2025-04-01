"""
Example script to plot q-Gaussian functions for various parameters
"""
from fma_ions import Fit_Functions
import numpy as np
import matplotlib.pyplot as plt


# Instantiate the class
fit_func = Fit_Functions() 

# Define the x-axis range
x = np.linspace(-4, 4, 1000)

# Define the parameters for the q-Gaussians
params = [
    {'q': 0, 'beta': 1, 'A': 0.8, 'mu': 0, 'color': 'red'},
    {'q': 1, 'beta': 1, 'A': 0.8, 'mu': 0, 'color': 'black'},
    {'q': 2, 'beta': 1, 'A': 0.8, 'mu': 0, 'color': 'blue'},
    {'q': 2, 'beta': 3, 'A': 0.8, 'mu': 0, 'color': 'lime'},
]

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), constrained_layout=True)
#plt.title("q-Gaussian Distribution for Different q and β values")
ax.set_xlabel("u")
ax.set_ylabel("g(u)")

# Plot each q-Gaussian
for i, p in enumerate(params):
    y = fit_func.Q_Gaussian(x, mu=p['mu'], q=p['q'], beta=p['beta'], A=p['A'])
    ax.plot(x, y, label=f"q={p['q']}, β={p['beta']}", linewidth=2.0, color=p['color'])

# Add a legend
ax.legend(fontsize=14)

# Add grid
ax.grid(True, alpha=0.55)
fig.savefig('q_Gaussian_functions.png', dpi=350)


# Show the plot
plt.show()
