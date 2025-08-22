Examples
========

This section provides practical examples of using fma_ions for various accelerator physics studies.

Basic FMA Study
---------------

Analyze tune diffusion for SPS Pb ions:

.. code-block:: python

   import numpy as np
   from fma_ions import FMA, SPS_sequence_maker, BeamParameters_SPS

   # Setup SPS lattice
   sps = SPS_sequence_maker()
   line, twiss = sps.load_xsuite_line_and_twiss()
   
   # Initialize beam parameters
   beam_params = BeamParameters_SPS()
   
   # Create FMA instance
   fma = FMA(line, beam_params, n_turns=1200)
   
   # Generate particles in tune space
   n_particles = 1000
   particles = fma.generate_particles_on_grid(
       qx_min=26.0, qx_max=26.5,
       qy_min=26.0, qy_max=26.5,
       n_points=int(np.sqrt(n_particles))
   )
   
   # Track particles
   x_data, y_data = fma.track_particles(particles)
   
   # Perform FMA analysis
   tune_diffusion, qx_array, qy_array = fma.run_FMA(x_data, y_data)
   
   # Plot tune diagram with diffusion
   fma.plot_tune_diagram(qx_array, qy_array, tune_diffusion,
                         title="SPS Pb Ion FMA")

Space Charge Study
------------------

Compare tune diffusion with and without space charge:

.. code-block:: python

   from fma_ions import SPS_Flat_Bottom_Tracker
   import matplotlib.pyplot as plt

   # Case 1: Without space charge
   tracker_no_sc = SPS_Flat_Bottom_Tracker(
       install_SC_on_line=False,
       bunch_intensity=0  # No space charge
   )
   tracker_no_sc.track_SPS(n_turns=1000)
   
   # Case 2: With space charge
   tracker_with_sc = SPS_Flat_Bottom_Tracker(
       install_SC_on_line=True,
       bunch_intensity=1e8  # Typical intensity
   )
   tracker_with_sc.track_SPS(n_turns=1000)
   
   # Compare emittance evolution
   plt.figure(figsize=(10, 6))
   plt.subplot(121)
   plt.plot(tracker_no_sc.container.turn_list, 
            tracker_no_sc.container.eps_x, label='No SC')
   plt.plot(tracker_with_sc.container.turn_list, 
            tracker_with_sc.container.eps_x, label='With SC')
   plt.xlabel('Turn')
   plt.ylabel('Horizontal Emittance [m·rad]')
   plt.legend()
   
   plt.subplot(122)
   plt.plot(tracker_no_sc.container.turn_list, 
            tracker_no_sc.container.eps_y, label='No SC')
   plt.plot(tracker_with_sc.container.turn_list, 
            tracker_with_sc.container.eps_y, label='With SC')
   plt.xlabel('Turn')
   plt.ylabel('Vertical Emittance [m·rad]')
   plt.legend()
   
   plt.tight_layout()
   plt.show()

Tune Ripple Impact
------------------

Study the effect of 50 Hz power converter ripple:

.. code-block:: python

   from fma_ions import Tune_Ripple_SPS, SPS_Flat_Bottom_Tracker
   
   # Create baseline tracker
   tracker = SPS_Flat_Bottom_Tracker()
   
   # Initialize tune ripple
   ripple = Tune_Ripple_SPS(
       frequency=50.0,  # Hz
       amplitude_x=1e-4,  # Horizontal tune ripple
       amplitude_y=1e-4   # Vertical tune ripple
   )
   
   # Apply ripple during tracking
   for turn in range(1000):
       # Track one turn
       tracker.track_one_turn()
       
       # Apply tune ripple
       ripple_value = ripple.get_ripple_value(turn)
       tracker.apply_tune_shift(ripple_value)
   
   # Analyze tune spread
   tracker.plot_tune_evolution()

Dynamic Aperture Scan
---------------------

Systematic dynamic aperture study:

.. code-block:: python

   from fma_ions import FMA, BeamParameters_SPS
   import numpy as np
   
   def dynamic_aperture_scan(amplitudes, n_angles=8):
       """Scan dynamic aperture at different amplitudes."""
       
       results = {'amplitude': [], 'survival_fraction': []}
       
       for amplitude in amplitudes:
           survival_count = 0
           
           for angle in np.linspace(0, np.pi/2, n_angles):
               # Generate particles at specific amplitude and angle
               x_norm = amplitude * np.cos(angle)
               y_norm = amplitude * np.sin(angle)
               
               # Convert normalized coordinates to physical
               beam_params = BeamParameters_SPS()
               x_phys = x_norm * np.sqrt(beam_params.epsilon_x)
               y_phys = y_norm * np.sqrt(beam_params.epsilon_y)
               
               # Track particle
               fma = FMA()
               particles = fma.generate_single_particle(x_phys, y_phys)
               
               try:
                   x_data, y_data = fma.track_particles(particles, n_turns=1000)
                   
                   # Check if particle survived
                   if not np.any(np.isnan(x_data)) and not np.any(np.isnan(y_data)):
                       survival_count += 1
               except:
                   pass
           
           survival_fraction = survival_count / n_angles
           results['amplitude'].append(amplitude)
           results['survival_fraction'].append(survival_fraction)
           
           print(f"Amplitude {amplitude:.2f}σ: {survival_fraction:.1%} survival")
       
       return results
   
   # Run scan
   amplitudes = np.linspace(1, 10, 10)
   da_results = dynamic_aperture_scan(amplitudes)
   
   # Plot results
   plt.figure(figsize=(8, 6))
   plt.plot(da_results['amplitude'], da_results['survival_fraction'], 'o-')
   plt.xlabel('Amplitude [σ]')
   plt.ylabel('Survival Fraction')
   plt.title('Dynamic Aperture')
   plt.grid(True)
   plt.show()

Batch Processing with HTCondor
------------------------------

Submit multiple jobs for parameter scans:

.. code-block:: python

   from fma_ions import Submitter
   import numpy as np
   
   # Define parameter ranges
   intensities = [5e7, 1e8, 2e8, 3e8]  # Bunch intensities
   n_turns = 5000
   
   # Create submitter
   submitter = Submitter(
       job_flavour="longlunch",  # HTCondor job flavour
       output_dir="./fma_scan_results"
   )
   
   # Submit jobs for each intensity
   for i, intensity in enumerate(intensities):
       job_name = f"fma_intensity_scan_{i:02d}"
       
       submitter.submit_job(
           job_name=job_name,
           script_template="sps_fma_template.py",
           parameters={
               'bunch_intensity': intensity,
               'n_turns': n_turns,
               'output_file': f"fma_results_{intensity:.0e}.pkl"
           }
       )
       
       print(f"Submitted job {job_name} for intensity {intensity:.1e}")

Advanced: Multi-Species Comparison
----------------------------------

Compare FMA for different ion species:

.. code-block:: python

   from fma_ions import BeamParameters_SPS_Pb, BeamParameters_SPS_Oxygen
   
   species_configs = {
       'Pb': BeamParameters_SPS_Pb(),
       'O': BeamParameters_SPS_Oxygen()
   }
   
   results = {}
   
   for species_name, beam_params in species_configs.items():
       print(f"Analyzing {species_name} ions...")
       
       # Setup FMA
       fma = FMA(line, beam_params)
       
       # Track particles
       particles = fma.generate_particles_on_grid(n_points=50)
       x_data, y_data = fma.track_particles(particles)
       
       # Analyze
       d, qx, qy = fma.run_FMA(x_data, y_data)
       
       results[species_name] = {
           'tune_diffusion': d,
           'qx': qx,
           'qy': qy
       }
   
   # Compare results
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   
   for i, (species, data) in enumerate(results.items()):
       ax = axes[i]
       scatter = ax.scatter(data['qx'], data['qy'], c=data['tune_diffusion'], 
                           cmap='viridis', vmin=-6, vmax=-2)
       ax.set_xlabel('Qx')
       ax.set_ylabel('Qy')
       ax.set_title(f'{species} Ions')
       plt.colorbar(scatter, ax=ax, label='log₁₀(d)')
   
   plt.tight_layout()
   plt.show()

Performance Optimization
------------------------

Tips for large-scale studies:

.. code-block:: python

   # Use GPU acceleration when available
   import xtrack as xt
   
   try:
       context = xt.ContextCupy()  # GPU context
       print("Using GPU acceleration")
   except:
       context = xt.ContextCpu()   # Fallback to CPU
       print("Using CPU")
   
   # Batch particle generation for efficiency
   def efficient_fma_scan(n_particles=10000, batch_size=1000):
       """Process particles in batches to manage memory."""
       
       results = []
       
       for batch_start in range(0, n_particles, batch_size):
           batch_end = min(batch_start + batch_size, n_particles)
           batch_particles = fma.generate_particles(batch_end - batch_start)
           
           # Track batch
           x_batch, y_batch = fma.track_particles(batch_particles)
           d_batch, qx_batch, qy_batch = fma.run_FMA(x_batch, y_batch)
           
           results.append({
               'tune_diffusion': d_batch,
               'qx': qx_batch,
               'qy': qy_batch
           })
           
           print(f"Completed batch {batch_start//batch_size + 1}")
       
       return results
