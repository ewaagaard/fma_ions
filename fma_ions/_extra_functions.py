"""
Container class for miscellaneous functions that might be useful for the future
"""
from dataclasses import dataclass
import numpy as np
import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo

class ExtraFunctions:

    def introduce_beta_beat(self, line : xt.Line, twiss : xt.TwissTable, beta_beat : float) -> xt.Line:
        """Method to introduce quadrupolar error"""

        # Create knobs controlling all quads
        ltab = line.get_table()
        line.vars['k1l.qf'] = 0
        line.vars['k1l.qd'] = 0

        qftab = ltab.rows['qf.*']
        for i, nn in enumerate(qftab.name):
            if qftab.element_type[i] == 'Multipole':
                line.element_refs[nn].knl[1] = line.vars['k1l.qf']

        qdtab = ltab.rows['qd.*']
        for i, nn in enumerate(qdtab.name):
            if qdtab.element_type[i] == 'Multipole':
                line.element_refs[nn].knl[1] = line.vars['k1l.qd']

        # First add extra knob for the quadrupole
        line.vars['kk_QD'] = 0
        line.element_refs['qd.63510..1'].knl[1] = line.vars['kk_QD']
        
        # Find where this maximum beta function occurs
        betx_max_loc = twiss.rows[np.argmax(twiss.betx)].name[0]
        betx_max = (1 + beta_beat) * twiss.rows[np.argmax(twiss.betx)].betx[0]

        # Rematch the tunes with the knobs
        line.match(
            vary=[
                xt.Vary('k1l.qf', step=1e-8),
                xt.Vary('k1l.qd', step=1e-8),
                xt.Vary('kk_QD', step=1e-8),  #vary knobs and quadrupole simulatenously 
            ],
            targets = [
                xt.Target('qx', self.qx0, tol=1e-7),
                xt.Target('qy', self.qy0, tol=1e-7),
                xt.Target('betx', value=betx_max, at=betx_max_loc, tol=1e-7)
            ])
        
        return line 
    
    
    def generate_line(self, 
                  add_non_linear_magnet_errors=False, 
                  add_aperture=True,
                  beta_beat=None, 
                  harmonic_nb = 4653,
                  Qy_frac: int = 25,
                  minimum_aperture_to_remove=0.025,
                  deferred_expressions=False,
                  )->xt.Line:
        """

        Generate SPS lines with fixed transverse aperture, longitudinally limitRect for bucket, beta-beat
        and magnet errors

        Parameters:
        -----------
        Qy_frac : int
            fractional part of vertical tune
        add_non_linear_magnet_errors : bool
            whether to add line with non-linear chromatic errors
        add_aperture : bool
            whether to include aperture for SPS
        beta_beat : float
            relative beta beat, i.e. relative difference between max beta function and max original beta function
        harmonic_nb : int
            harmonic used for SPS RF system
        ibs_step : int
            turn interval at which to recalculate IBS growth rates
        Qy_frac : int
            fractional part of vertical tune, e.g. "19" for 26.19
        minimum_aperture_to_remove : float 
            minimum threshold of horizontal SPS aperture to remove, default is 0.025 (can also be set to None)
            as faulty IPM aperture has 0.01 m, which is too small
        deferred_expressions : bool
            whether to use deferred expressions while importing madx sequence into xsuite

        Returns:
        --------
        xt.Line, str
        """
        # Update vertical tune if changed
        self.qy0 = int(self.qy0) + Qy_frac / 100

        # Get SPS Pb line - with aperture and non-linear magnet errors if desired
        sps = SPS_sequence_maker()
        line, _ = sps.load_xsuite_line_and_twiss(Qy_frac=Qy_frac, add_aperture=add_aperture, beta_beat=beta_beat,
                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors)
                
        if minimum_aperture_to_remove is not None:
            line = sps.remove_aperture_below_threshold(line, minimum_aperture_to_remove)

        # Add longitudinal limit rectangle - to kill particles that fall out of bucket
        bucket_length = line.get_length()/harmonic_nb
        line.unfreeze() # if you had already build the tracker
        line.append_element(element=xt.LongitudinalLimitRect(min_zeta=-bucket_length/2, max_zeta=bucket_length/2), name='long_limit')

        # Generate name of file
        def_exp_str = '_deferred_exp' if deferred_expressions else ''
        BB_string = '_{}_percent_beta_beat'.format(int(beta_beat*100)) if beta_beat is not None else ''
        err_str = '_with_magnet_errors' if add_non_linear_magnet_errors else ''
        fname = 'SPS_2021_Pb_Qydot{}{}{}{}.json'.format(Qy_frac, def_exp_str, BB_string, err_str)

        return line, fname
    

    def save_lines_for_all_cases(self, output_folder : str ='lines', also_save_lines_with_deferred_expressions=True):
        """
        Generate lines for all cases of SPS flat bottom tracking: magnet errors, beta-beating, etc
        Used for instance on HTCondor where input file needs to be provided
        """
        os.makedirs(output_folder, exist_ok=True)

        # Load ideal lattice, and with BB + magnet errors
        line_ideal, f_ideal = self.generate_line(add_aperture=True, beta_beat=None, add_non_linear_magnet_errors=False)
        line_ideal_def_exp, f_ideal_def_exp = self.generate_line(add_aperture=True, beta_beat=None, add_non_linear_magnet_errors=False,
                                                           deferred_expressions=True)
        line_bb, f_bb = self.generate_line(add_aperture=True, beta_beat=0.1, add_non_linear_magnet_errors=True)
        line_bb_def_exp, f_bb_def_exp = self.generate_line(add_aperture=True, beta_beat=0.1, add_non_linear_magnet_errors=True,
                                                           deferred_expressions=True)
        line_ideal_dot19, f_ideal_dot19 = self.generate_line(Qy_frac=19, add_aperture=True, beta_beat=None, add_non_linear_magnet_errors=False)
        line_ideal_def_exp_dot19, f_ideal_def_exp_dot19 = self.generate_line(Qy_frac=19, add_aperture=True, beta_beat=None, add_non_linear_magnet_errors=False,
                                                           deferred_expressions=True)
        line_bb_dot19, f_bb_dot19 = self.generate_line(Qy_frac=19, add_aperture=True, beta_beat=0.1, add_non_linear_magnet_errors=True)
        line_bb_def_exp_dot19, f_bb_def_exp_dot19 = self.generate_line(Qy_frac=19, add_aperture=True, beta_beat=0.1, add_non_linear_magnet_errors=True,
                                                           deferred_expressions=True)
        

        lines = [line_ideal, line_bb, line_ideal_dot19, line_bb_dot19]
        lines_def_exp = [line_ideal_def_exp, line_bb_def_exp, line_ideal_def_exp_dot19, line_bb_def_exp_dot19]
        str_names = [f_ideal, f_bb, f_ideal_dot19, f_bb_dot19]
        str_names_def_exp = [f_ideal_def_exp, f_bb_def_exp, f_ideal_def_exp_dot19, f_bb_def_exp_dot19]

        # Dump lines to json files
        for i, line in enumerate(lines):
            sps_fname = f'{output_folder}/{str_names[i]}'
            print(f'Saving {sps_fname}')
            with open(sps_fname, 'w') as fid:
                json.dump(line.to_dict(), fid, cls=xo.JEncoder)
        
        # Also save strings to 
        with open(f'{output_folder}/line_names.txt', 'w') as outfile:
            outfile.write('\n'.join(str(i) for i in str_names))
        
        if also_save_lines_with_deferred_expressions:
            for i, line in enumerate(lines_def_exp):
                sps_fname_def_exp = f'{output_folder}/{str_names_def_exp[i]}'
                print(f'Saving {sps_fname_def_exp}')
                with open(sps_fname_def_exp, 'w') as fid:
                    json.dump(line.to_dict(), fid, cls=xo.JEncoder)
            
            # Also save strings to 
            with open(f'{output_folder}/line_names_def_exp.txt', 'w') as outfile:
                outfile.write('\n'.join(str(i) for i in str_names_def_exp))


    def track_SPS_with_prepared_line(self, line : xt.Line,
                                        which_context='gpu',
                                        beamParams=None,
                                        install_SC_on_line=True, 
                                        SC_mode='frozen',
                                        use_Gaussian_distribution=True,
                                        apply_kinetic_IBS_kicks=False,
                                        ibs_step = 50,
                                        ):
        """
        Run full tracking at SPS flat bottom with prepared input line, returning pandas dataframe
        
        Parameters:
        ----------
        line : xt.Line
            input line on which to do tracking
        which_context : str
            'gpu' or 'cpu'
        Qy_frac : int
            fractional part of vertical tune
        beamParams : dataclass
            container of exn, eyn, Nb and sigma_z. Default 'None' will load nominal SPS beam parameters 
        install_SC_on_line : bool
            whether to install space charge
        SC_mode : str
            type of space charge - 'frozen' (recommended), 'quasi-frozen' or 'PIC'
        use_Gaussian_distribution : bool
            whether to use Gaussian particle distribution for tracking
        add_kinetic_IBS_kicks : bool
            whether to apply kinetic kicks from xibs 
        ibs_step : int
            turn interval at which to recalculate IBS growth rates

        Returns:
        --------
        pd.DataFrame
        """
        # Initial settings for GPU device 
        gpu_device = 0

        # If specific beam parameters are not provided, load default SPS beam parameters
        if beamParams is None:
            beamParams = BeamParameters_SPS
        print('Beam parameters:', beamParams)

        # Select relevant context
        if which_context=='gpu':
            context = xo.ContextCupy()
        elif which_context=='cpu':
            context = xo.ContextCpu(omp_num_threads='auto')
        else:
            raise ValueError('Context is either "gpu" or "cpu"')

        line.build_tracker(_context=context)
        twiss = line.twiss()

        # Generate particles object to track    
        particles = self.generate_particles(line=line, context=context, use_Gaussian_distribution=use_Gaussian_distribution,
                                            beamParams=beamParams)

        # Initialize the dataclasses and store the initial values
        tbt = Records.init_zeroes(self.num_turns)
        tbt.update_at_turn(0, particles, twiss)

        ######### IBS kinetic kicks #########
        if apply_kinetic_IBS_kicks:
            beamparams = BeamParameters.from_line(line, n_part=beamParams.Nb)
            opticsparams = OpticsParameters.from_line(line) # read from line without space  charge
            IBS = KineticKickIBS(beamparams, opticsparams)
            kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
            print(kinetic_kick_coefficients)

        # Install SC and build tracker
        if install_SC_on_line:
            fma_sps = FMA()
            line = fma_sps.install_SC_and_get_line(line, beamParams, mode=SC_mode, optimize_for_tracking=True, context=context)
            print('Installed space charge on line\n')

        # Start tracking 
        for turn in range(1, self.num_turns):
            
            if turn % self.turn_print_interval == 0:
                print('Tracking turn {}'.format(turn))            

            ########## IBS -> Potentially re-compute the ellitest_parts integrals and IBS growth rates #########
            if apply_kinetic_IBS_kicks and ((turn % ibs_step == 0) or (turn == 1)):
                
                # We compute from values at the previous turn
                kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
                print(
                    "\n" + "=" * 60 + "\n",
                    f"Turn {turn:d}: re-computing growth rates and kick coefficients\n",
                    kinetic_kick_coefficients,
                    "\n" + "=" * 60,
                )
                
            ########## ----- Apply IBS Kick if desired ----- ##########
            if apply_kinetic_IBS_kicks:
                IBS.apply_ibs_kick(particles)
            
            # ----- Track and update records for tracked particles ----- #
            line.track(particles, num_turns=1)
            tbt.update_at_turn(turn, particles, twiss)

        # Make parquet file from dictionary
        tbt_dict = tbt.to_dict()
        df = pd.DataFrame(tbt_dict)
        
        return df