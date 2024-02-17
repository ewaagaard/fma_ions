"""
Class container for methods to track xpart particle objects at flat bottom
- for SPS
- choose context (GPU, CPU) and additional effects: SC, IBS, tune ripples
"""
from dataclasses import dataclass
import numpy as np
import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo

from .sequence_classes_ps import PS_sequence_maker, BeamParameters_PS
from .sequence_classes_sps import SPS_sequence_maker, BeamParameters_SPS
from .fma_ions import FMA

class SPS_Flat_Bottom_Tracker:
    """
    Container to track particles 
    """

    def _bunch_length(parts: xp.Particles) -> float:
        """Helper function to return bunch length"""
        return np.std(parts.zeta[parts.state > 0])


    def _sigma_delta(parts: xp.Particles) -> float:
        return np.std(parts.delta[parts.state > 0])


    def _geom_epsx(parts: xp.Particles, twiss: xt.TwissTable) -> float:
        """
        We index dx and betx at 0 which corresponds to the beginning / end of
        the line, since this is where / when we will be applying the kicks.
        """
        sigma_x = np.std(parts.x[parts.state > 0])
        sig_delta = _sigma_delta(parts)
        return (sigma_x**2 - (twiss["dx"][0] * sig_delta) ** 2) / twiss["betx"][0]


    def _geom_epsy(parts: xp.Particles, twiss: xt.TwissTable) -> float:
        """
        We index dy and bety at 0 which corresponds to the beginning / end of
        the line, since this is where / when we will be applying the kicks.
        """
        sigma_y = np.std(parts.y[parts.state > 0])
        sig_delta = _sigma_delta(parts)
        return (sigma_y**2 - (twiss["dy"][0] * sig_delta) ** 2) / twiss["bety"][0]

    def track_SPS(self, sc_mode='frozen'):

        # Get SPS Pb line with deferred expressions
        sps = SPS_sequence_maker()
        line, twiss = sps.load_SPS_line_and_twiss(Qy_frac=Qy_frac, add_aperture=add_aperture, beta_beat=None,
                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors)
        
        # Install SC, track particles and observe tune diffusion
        
        # Add beta-beat if desired 
        if beta_beat is not None:
            sps_seq = SPS_sequence_maker()
            line = sps_seq.generate_xsuite_seq_with_beta_beat(beta_beat=beta_beat, line=line, plane=plane_beta_beat)

        if install_SC_on_line:
            fma_sps = FMA()
            line = fma_sps.install_SC_and_get_line(line, BeamParameters_SPS(), optimize_for_tracking=False)
            print('Installed space charge on line\n')
        kqf_vals, kqd_vals, turns = self.load_k_from_xtrack_matching(dq=dq, plane=plane)

        line = self.install_SC_and_get_line(line0, beamParams)
        particles = self.generate_particles(line, beamParams, make_single_Jy_trace)
        x, y = self.track_particles(particles, line)


        
    def track_particles(self, particles, line, save_tbt_data=True, context='gpu'):
        """
        Track particles through lattice with space charge elments installed
        - save emittances, BL, sigma_delta and particle states every turns
        
        Parameters:
        ----------
        particles : xpart.particles
            particles object from xpart
        line : xtrack.line
            xsuite line to track through
        save_tbt: bool
            whether to save turn-by-turn data from tracking
        context : str
            'gpu' or 'cpu'
            
        Returns:
        -------
        x, y - numpy.ndarrays
            arrays containing turn-by-turn data coordinates
        """          

        # Select relevant context
        if context=='gpu':
            context = xo.ContextCupy()
        elif context=='cpu':
            context = xo.ContextCpu()
        else:
            raise ValueError('Context is either "gpu" or "cpu"')

        #### TRACKING #### 
        # Track the particles and return turn-by-turn coordinates
        x = np.zeros([self.num_part, self.num_turns]) 
        y = np.zeros([self.num_part, self.num_turns])
        exn = np.zeros([self.num_part, self.num_turns]) 
        eyn = np.zeros([self.num_part, self.num_turns])
        Nb = np.zeros([self.num_part, self.num_turns])
        sig_delta = np.zeros([self.num_part, self.num_turns])
        bl = np.zeros([self.num_part, self.num_turns])
        state = np.zeros([self.num_part, self.num_turns])
        tw = line.twiss()
        
        print('\nStarting tracking...')
        i = 0
        for turn in range(self.num_turns):
            if i % 20 == 0:
                print('Tracking turn {}'.format(i))
        
            if context=='gpu':
                f = 2
            elif context=='cpu':

                # Record TBT data and calculate emittance - only for particles still alive
                x[:, i] = particles.x
                y[:, i] = particles.y

                sig_x = np.std(particles.x[particles.state > 0])
                sig_y = np.std(particles.y[particles.state > 0])
                sig_delta = np.std(particles.delta[particles.state > 0])
                sig_delta[:, i] = sig_delta

                exn[:, i] = (sig_x**2 - (tw['dx'][0] * sig_delta)**2) / tw['betx'][0]
                eyn[:, i] = sig_y**2 / tw['bety'][0] 
                
                bl[:, i] = np.std(particles.zeta[particles.state > 0])
                
        
            # Track the particles
            line.track(particles)
            i += 1
        
        print('Finished tracking.\n')
        
        # Set particle trajectories of dead particles that got lost in tracking
        self._kill_ind = particles.state < 1
        self._kill_ind_exists = True
        
        if save_tbt_data:
            os.makedirs(self.output_folder, exist_ok=True)
            np.save('{}/x.npy'.format(self.output_folder), x)
            np.save('{}/y.npy'.format(self.output_folder), y)
            np.save('{}/px.npy'.format(self.output_folder), px)
            np.save('{}/py.npy'.format(self.output_folder), py)
            np.save('{}/x0_norm.npy'.format(self.output_folder), self._x_norm)
            np.save('{}/y0_norm.npy'.format(self.output_folder), self._y_norm)
            np.save('{}/state.npy'.format(self.output_folder), self._kill_ind)
            print('Saved tracking data.')
        

        return x, y