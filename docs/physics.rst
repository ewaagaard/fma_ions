Physics Background
==================

Frequency Map Analysis Theory
-----------------------------

Frequency Map Analysis (FMA) is a powerful technique for studying nonlinear dynamics in particle accelerators. It detects chaotic motion by analyzing tune diffusion over different time windows.

Tune Diffusion
~~~~~~~~~~~~~~

The tune diffusion parameter :math:`d` quantifies the stability of particle motion:

.. math::
   d = \log_{10} \sqrt{(Q_{x,2} - Q_{x,1})^2 + (Q_{y,2} - Q_{y,1})^2}

Where:
- :math:`Q_{x,y,1}`: Tunes calculated from first block of turns (e.g., turns 1-600)
- :math:`Q_{x,y,2}`: Tunes calculated from second block of turns (e.g., turns 601-1200)

**Interpretation:**
- :math:`d < -5`: Regular motion (stable)
- :math:`d > -3`: Chaotic motion (unstable)
- :math:`-5 < d < -3`: Intermediate region

Resonance Identification
~~~~~~~~~~~~~~~~~~~~~~~~

FMA helps identify which resonances drive particle losses:

**Linear resonances:** :math:`m Q_x + n Q_y = p`

**Nonlinear resonances:** :math:`m Q_x + n Q_y = p` where :math:`m + n > 2`

Common resonances in ion accelerators:
- Third-order: :math:`3Q_x = p`, :math:`Q_x + 2Q_y = p`
- Fourth-order: :math:`2Q_x + 2Q_y = p`
- Higher-order coupling resonances

Ion Beam Physics
----------------

Space Charge Effects
~~~~~~~~~~~~~~~~~~~~~

Ion beams have significantly stronger space charge than proton beams due to:
- Higher charge-to-mass ratio (:math:`Q/A`)
- Lower particle velocity (:math:`\beta`)

The space charge tune shift scales as:

.. math::
   \Delta Q \propto \frac{Q^2 N_b}{\beta^2 \gamma^3 \epsilon}

Where:
- :math:`Q`: Ion charge state
- :math:`N_b`: Bunch intensity
- :math:`\beta, \gamma`: Relativistic factors
- :math:`\epsilon`: Beam emittance

Intra-Beam Scattering (IBS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IBS causes:
- Emittance growth in all planes
- Momentum spread increase
- Bunch length evolution

IBS growth rates depend on:
- Beam density
- Coulomb logarithm
- Lattice functions

Tune Ripple
~~~~~~~~~~~

Power converter ripple introduces time-varying tune modulation:

.. math::
   Q(t) = Q_0 + \Delta Q \sin(2\pi f_{ripple} t + \phi)

**50 Hz ripple** from power grid is particularly problematic as it can:
- Drive resonances
- Enhance tune diffusion
- Reduce dynamic aperture

CERN Accelerator Complex
-------------------------

SPS (Super Proton Synchrotron)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Key parameters for Pb ions:**
- Energy: 177 GeV/u (flat bottom) → 6.5 TeV/u (flat top)
- Circumference: 6.9 km
- Typical bunch intensity: :math:`1-3 \times 10^8` ions
- Space charge parameter: :math:`\Delta Q_{sc} \sim 0.01-0.05`

**Physics challenges:**
- Strong space charge at injection
- IBS-driven emittance growth
- RF capture efficiency
- Beam-beam effects (when used as LHC injector)

PS (Proton Synchrotron)
~~~~~~~~~~~~~~~~~~~~~~~

**Pb ion injection from LEIR:**
- Energy: 5.9 MeV/u → 5.9 GeV/u
- Bunch intensity: :math:`\sim 5 \times 10^7` ions
- Strong space charge effects
- Longitudinal bunch splitting

LEIR (Low Energy Ion Ring)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Ion accumulation and cooling:**
- Energy: 4.2 MeV/u
- Electron cooling
- Multi-turn injection from Linac3
- Bunch formation and preparation

Simulation Techniques
---------------------

Tracking Methods
~~~~~~~~~~~~~~~~

**Symplectic tracking** preserves phase space area:
- Essential for long-term stability studies
- xsuite provides high-performance symplectic integrators

**Space charge models:**
- Frozen space charge: Fast, suitable for short-term studies
- Adaptive space charge: Self-consistent, computationally intensive

**GPU acceleration** enables:
- Tracking millions of particles
- Thousands of turns
- Statistical studies with many seeds

Analysis Workflow
~~~~~~~~~~~~~~~~~

1. **Lattice setup**: Load accelerator model
2. **Beam generation**: Create particle distributions
3. **Tracking**: Simulate particle motion
4. **FMA analysis**: Calculate tune diffusion
5. **Visualization**: Generate tune diagrams and loss plots

**Best practices:**
- Use sufficient statistics (>1000 particles)
- Track for adequate time (>1000 turns)
- Consider multiple seeds for statistical analysis
- Validate against measurements when available
