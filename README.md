# Frequency Map Analysis for ions

Non-linear effects can enhance chaotic motion and limit the performance of accelerators. Frequency Map Analysis (FMA) is a method that probes tune diffusion from particle tracking simulations and identify resonances that are excited. FMA can be used to detect chaotic behaviour in particle acceelerators, by looking at the tune diffusion $d$ 

$$d =  \log \sqrt{ (Q_{x, 2} - Q_{x, 1})^2 + (Q_{y, 2} - Q_{y, 1})^2}$$

where $Q_{x, y, 1}$ is the particle tunes in the first number of turns (e.g. first 600 turns) and $Q_{x, y, 2}$ is the tunes of the subsequent turn block (e.g. the next 600 turns).  

This repository contains FMA classes for ion beams in the CERN accelerator complex, but can also be used for other protons and other set-ups.  

![FMA_plot_SPS](https://github.com/ewaagaard/fma_ions/assets/68541324/ed6676c6-9812-486e-81c6-6f6ea0b411de)

### Quick set-up

When using Python for scientific computing, it is important to be aware of dependencies and compatibility of different packages. This guide gives a good explanation: [Python dependency manager guide](https://aaltoscicomp.github.io/python-for-scicomp/dependencies/#dependency-management). An isolated environment allows installing packages without affecting the rest of your operating system or any other projects. A useful resource to handle virtual environments is [Anaconda](https://www.anaconda.com/) (or its lighter version Miniconda), when once installed has many useful commands of which many can be found in the [Conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf). A guide to manage conda environments is found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 

To directly start using `fma_ions`, create an isolated virtual environment with conda and perform a local install to use the `fma_ions`. Run in the terminal to clone the repository, install a virtual environment (with all requirements) and then a local editable install of `fma_ions`:

```
git clone https://github.com/ewaagaard/fma_ions.git
conda create --name fma_env python=3.11.7
conda activate fma_env
python -m pip install -r venvs/requirements.txt
cd ..
python -m pip install -e fma_ions
```

#### Cloning repos for PS and SPS sequences

The `fma_ions` relies on sequence generators for PS and SPS ions, which requires MADX sequence files from [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) from [acc-models](https://gitlab.cern.ch/acc-models). Clone these repositories as submodules inside the `fma_ions`:
```
cd fma_ions/data/
git clone https://gitlab.cern.ch/acc-models/acc-models-sps.git
git clone https://gitlab.cern.ch/acc-models/acc-models-ps.git
```
If this command is executed correctly, two repositories `acc-models-ps` and `acc-models-sps` should appear inside the `data` folder with content. 

#### GPU support

Tracking many particles for millions of turns can take a long time. Although the default context for most functions is with CPU, most of them also contain support for GPUs for faster particle tracking. If you machine supports GPU usage, check out [Xsuite GPU/Multithreading support](https://xsuite.readthedocs.io/en/latest/installation.html#gpu-multithreading-support). The `cupy` should already be installed from `venvs/requirements.txt`, but it can be useful to run the following lines again:

```
conda install mamba -n base -c conda-forge
pip install cupy-cuda11x
mamba install cudatoolkit=11.8.0
```

## FMA class

The `FMA` class contains all the methods and imports all helper functions needed to track particle objects and analyze the tune diffusion. The class is instantiated by
```
import fma_ions
fma = fma_ions.FMA() 
```
Optional arguments can be added, such as specific output destinations for turn-by-turn (TBT) data and plots, number of turns to track, shape of particle object, longitudinal position offset `z0` and so on. Arbitrary [Xsuite](https://xsuite.readthedocs.io/en/latest/) lines can be used, although helper classes provide standard PS and SPS lattice models. 

The FMA happens in several steps: 
1) first a line with space charge (SC) is returned, and the particle object to be tracked. Input is an Xsuite line (without space charge) and `beamParams`, which is a data class containing bunch intensity `Nb`, bunch length `sigma_z`, normalized emittances `exn` and `eyn`, and the integer tune of the accelerator `Q_int`, since this technique can only detect fractional tunes. An example structure for this `beamParams` class is found in `fma_ions.BeamParameters_SPS()`. 
2) TBT `x` and `y` data are returned, tracking for the specified attribute `num_turns` (default 1200) in the FMA class.
3) The diffusion coefficient `d` and individual particle tunes `Qx` and `Qy` are returned from the NAFF algorithm used in the FMA method
4) `Qx`, `Qy` and `d` can be plotted in the initial particle distribution and in the final tune diagram of the particles. 
```
line_with_SC, particles = fma.install_SC_and_generate_particles(line, beamParams)
x, y = fma.track_particles(particles, line_with_SC)
d, Qx, Qy = fma.run_FMA(x, y)

# Set point, e.g for SPS
Qh_set, Qv_set = 26.30, 26.25 

# Tune footprint range
plot_range  = [[26.0, 26.35], [26.0, 26.35]]

# Generate plots.
fma.plot_FMA(d, Qx, Qy, Qh_set, Qv_set,'SPS', plot_range)
fma.plot_initial_distribution(x, y, d, case_name='SPS')
```
Initial distributions provide useful insights where the tune diffusion happens inside the distribution. 

![SPS_Initial_distribution](https://github.com/ewaagaard/fma_ions/assets/68541324/35a343bf-fc7d-4215-a4a2-08fe2e43be19)

### Built-in sequences 

SPS and PS standard Pb lattice models from Xsuite are contained in helper data classes. These can be instantiated easily by using:
```
import fma_ions
fma_sps = fma_ions.FMA()
fma_sps.run_SPS()

fma_ps = fma_ions.FMA()
fma_ps.run_PS()
```
which call standard PS and SPS lattices, and perform the FMA described in the previous section with default parameters. If the tracking has already been done, plots can easily be generated from saved TBT data by using
```
fma_sps = fma_ions.FMA()
fma_sps.run_SPS(load_tbt_data=True)
```
### Custom beams and tunes

Custom beam intensities, tunes, charge states and masses can be provided. The `FMA` class rematches provided tunes and generates the desired beam

```
import fma_ions

# Initialize FMA object and intialize beam (mass is in atomic units)
fma_sps = fma_ions.FMA()
fma_sps.run_custom_beam_SPS(ion_type='O', m_ion=15.99, Q_SPS=8., Q_PS=4., qx=26.30, qy=26.19, Nb=82e8)
```
### Longitudinal position or momentum offset

To investigate tune diagram sidebands and effects of synchrotron oscillations on off-momentum particles, non-zero `z0` or `delta0` can be provided to generate the particle object. If a uniform beam is used, `n_linear` determines the resolution of the meshgrid in normalized coordinates $X$ and $Y$, so `n_linear=100` generates 10 000 particles unformly distributed up to a desired beam size (e.g. 10 $\sigma$). 
```
import fma_ions

fma_sps = fma_ions.FMA(z0=0.15, n_linear=200)
fma_sps.run_SPS()
```

![FMA_plot_SPS](https://github.com/ewaagaard/fma_ions/assets/68541324/d1d69eec-fc0d-4ccd-a34c-a1820cc6f604)

