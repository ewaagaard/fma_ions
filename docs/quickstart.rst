Quick Start Guide
=================

This guide gets you started with the basic functionality of fma_ions.

Basic FMA Analysis
------------------

.. code-block:: python

   from fma_ions import FMA, SPS_sequence_maker, BeamParameters_SPS
   import numpy as np
   
   # 1. Create SPS sequence
   sps = SPS_sequence_maker()
   line, twiss = sps.load_xsuite_line_and_twiss()
   
   # 2. Set up beam parameters  
   beam_params = BeamParameters_SPS()
   
   # 3. Initialize FMA
   fma = FMA(line, beam_params)
   
   # 4. Generate particles and track
   particles = fma.generate_particles()
   x, y = fma.track_particles(particles)
   
   # 5. Run FMA analysis
   d, Qx, Qy = fma.run_FMA(x, y)
   
   # 6. Plot results
   fma.plot_tune_diagram(Qx, Qy, d)

SPS Flat Bottom Tracking
-------------------------

For comprehensive SPS simulations with space charge:

.. code-block:: python

   from fma_ions import SPS_Flat_Bottom_Tracker
   
   # Initialize tracker with default Pb ion parameters
   tracker = SPS_Flat_Bottom_Tracker()
   
   # Track particles for specified number of turns
   tracker.track_SPS(n_turns=1000)
   
   # Access tracking data
   data = tracker.container
   
   # Generate output plots
   tracker.plot_tracking_data()

HTCondor Job Submission
-----------------------

Submit jobs to CERN computing cluster:

.. code-block:: python

   from fma_ions import Submitter
   
   # Create submitter instance
   submitter = Submitter(
       job_name="my_fma_job",
       output_dir="/path/to/output"
   )
   
   # Submit tracking job
   submitter.submit_tracking_job(
       n_turns=10000,
       n_particles=1000,
       accelerator="SPS"
   )

Tune Ripple Analysis
--------------------

Simulate power converter ripple effects:

.. code-block:: python

   from fma_ions import Tune_Ripple_SPS
   
   # Create tune ripple instance
   ripple = Tune_Ripple_SPS()
   
   # Generate 50 Hz ripple signal
   time_array, ripple_signal = ripple.generate_ripple(
       frequency=50.0,  # Hz
       amplitude=1e-4,  # relative tune variation
       duration=1.0     # seconds
   )
   
   # Apply to beam
   ripple.apply_to_beam(particles, ripple_signal)

Working with Different Accelerators
------------------------------------

**PS (Proton Synchrotron):**

.. code-block:: python

   from fma_ions import PS_sequence_maker, BeamParameters_PS
   
   ps = PS_sequence_maker()
   line, twiss = ps.load_xsuite_line_and_twiss()
   beam_params = BeamParameters_PS()

**LEIR (Low Energy Ion Ring):**

.. code-block:: python

   from fma_ions import LEIR_sequence_maker, BeamParameters_LEIR
   
   leir = LEIR_sequence_maker()
   line, twiss = leir.load_xsuite_line_and_twiss()
   beam_params = BeamParameters_LEIR()

Next Steps
----------

- Explore the `examples/` directory for more complex use cases
- Read the :doc:`physics` guide for background on FMA theory
- Check the :doc:`examples` for specific applications
- Review the API reference for detailed function documentation
