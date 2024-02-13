"""
Main module for sequence generator container classes for LEIR

An overview of LEIR nominal optics can be found here: https://acc-models.web.cern.ch/acc-models/leir/2021/scenarios/nominal/
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
import os 
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt
import xpart as xp

from scipy import constants
from cpymad.madx import Madx
import json

optics =  Path(__file__).resolve().parent.joinpath('../data/acc-models-leir').absolute()
sequence_path = Path(__file__).resolve().parent.joinpath('../data/leir_sequences').absolute()
#error_file_path = Path(__file__).resolve().parent.joinpath('../data/sps_sequences/magnet_errors').absolute()

@dataclass
class BeamParameters_LEIR:
    """Data Container for LEIR Pb default beam parameters"""
    Nb:  float = 10.0e8 #3.5e8
    sigma_z: float = 8.0
    exn: float = 0.4e-6
    eyn: float = 0.4e-6
    Qx_int: float = 1.
    Qy_int: float = 2.
    
    
@dataclass
class LEIR_sequence_maker:
    """ 
    Data class to generate Xsuite line from SPS optics repo, selecting
    - qx0, qy0: horizontal and vertical tunes (from acc-model nominal at flat bottom)
    - dq1, dq2: X and Y chroma values 
    - m_ion: ion mass in atomic units
    - optics: absolute path to optics repository -> cloned from https://gitlab.cern.ch/acc-models
    """
    qx0: float = 1.82
    qy0: float = 2.72
    dq1: float = -0.02366845
    dq2: float = -0.00541004
    # Default SPS PB ION CHROMA VALUES: not displayed on acc-model, extracted from PTC Twiss 
    
    # Define beam type - default is Pb
    ion_type: str = 'Pb'
    seq_name: str = 'nominal'
    seq_folder: str = 'leir'
    Brho_LEIR_extr: float = 4.8 # [T] - magnetic field in PS for Pb ions, from Heiko Damerau
    m_ion: float = 207.98
    Q_LEIR: int = 54 # default charge state for Pb
    A : int = 208 # default mass number for Pb
    E_kin_ev_per_A_LEIR_inj = 4.2e6 # kinetic energy in eV per nucleon in LEIR before RF capture, same for all species

    
    def load_madx(self, make_thin=True, add_aperture=False):
        """
        Loads default LEIR Pb sequence at flat bottom. 
        
        Parameters:
        -----------
        make_thin : bool
            whether to slice the sequence or not
        add_aperture : bool
            whether to call aperture files 
        
        Returns: 
        --------    
        madx : madx instance with LEIR sequence    
        """
        madx = Madx()
        madx.call("{}/_scripts/macros.madx".format(optics))
        madx.call("{}/new_leir_seq.seq".format(optics))
        madx.call('{}/scenarios/nominal/1_flat_bottom/leir_fb_nominal.str'.format(optics))

        if add_aperture:
            madx.call("{}/leir.dbx".format(optics))

        # Load beam parameters from injection energy
        self.m_in_eV, self.gamma_LEIR_inj, self.p_LEIR_extr = self.generate_LEIR_beam()

        madx.input(" \
        Beam, particle=ion, mass={}, charge={}, gamma = {}, sequence='leir'; \
        exn={}, eyn={};\
        DPP:=BEAM->SIGE*(BEAM->ENERGY/BEAM->PC)^2;  \
        ".format(self.m_in_eV/1e9, self.Q_LEIR, self.gamma_LEIR_inj,
                        BeamParameters_LEIR.exn, BeamParameters_LEIR.eyn))   # convert mass to GeV/c^2

        # Slice the sequence
        if make_thin:
            n_slice_per_element = 5
            madx.command.select(flag='MAKETHIN', slice=n_slice_per_element, thick=False)
            madx.command.makethin(sequence='leir', MAKEDIPEDGE=True)  

        # Global correction of the coupling introduced by the electron cooler
        madx.input('''
                use, sequence=LEIR;
                exec, global_correction;;
                ''')
                
        madx.command.save(sequence='leir', file='{}/LEIR_2021_Pb_ions_4D.seq'.format(sequence_path), beam=True)

        return madx
        

    def generate_LEIR_beam(self):
        """
        Generate correct injection parameters for LEIR beam.
        At LEIR injection, all particles are presumably injected with 4.2 MeV/u
        
        Returns:
        -------
        ion rest mass in eV, beam momentum in eV/c at SPS injection 
        """
        # Calculate gamma at injection and extraction
        m_in_eV = self.m_ion * constants.physical_constants['atomic mass unit-electron volt relationship'][0]   # 1 Dalton in eV/c^2 -- atomic mass unit
        gamma_LEIR_inj = (m_in_eV + self.E_kin_ev_per_A_LEIR_inj * self.A) / m_in_eV
        p_LEIR_extr = 1e9 * (self.Brho_LEIR_extr * self.Q_LEIR) / 3.3356 # in  [eV/c], if q is number of elementary charges

        print('\nBeam: gamma = {:.5f}, m_ion = {:3e} eV\n'.format(gamma_LEIR_inj, m_in_eV))

        return m_in_eV, gamma_LEIR_inj, p_LEIR_extr
        

    def load_xsuite_line_and_twiss(self, beta_beat=None, save_new_xtrack_line=True,
                                   deferred_expressions=False):
        """
        Method to load pre-generated LEIR lattice files for Xsuite, or generate new if does not exist
        
        Parameters:
        -----------
        beta_beat : float
            relative beta beat, i.e. relative difference between max beta function and max original beta function
        deferred_expressions : bool
            whether to use deferred expressions while importing madx sequence into xsuite
        
        Returns:
        -------
        xsuite line
        twiss - twiss table from xtrack 
        """
        
        # Load LEIR line, otherwise generate new one
        leir_fname = '{}/LEIR_2021_Pb_ions.json'.format(sequence_path)
        print('Attempting to load {}'.format(leir_fname))
        
        try:         
            leir_line = xt.Line.from_json(leir_fname)
            print('\nLoaded LEIR json sequence file\n')
        except FileNotFoundError:
            print('\nDid not find LEIR json sequence file - generating new!\n')
            leir_line = self.generate_xsuite_seq(deferred_expressions=False, add_aperture=False, save_xsuite_seq=True)
        
        twiss_leir = leir_line.twiss() 
        
        return leir_line, twiss_leir
        

    def generate_xsuite_seq(self, save_madx_seq=False, 
                            save_xsuite_seq=False, 
                            return_xsuite_line=True, voltage=3.2e3,
                            deferred_expressions=False,
                            add_aperture=False):
        """
        Load MADX line, match tunes and chroma, add RF and generate Xsuite line
        
        Parameters:
        -----------
        save_madx_seq : bool
            save madx sequence to directory 
        save_xsuite_seq : bool
            save xtrack sequence to directory  
        return_xsuite_line : bool
            return generated xtrack line
        voltage : float 
            RF voltage, from LSA setting (search Obisidan note 'RF in LEIR')
        add_non_linear_magnet_errors : bool
            whether to add line with non-linear chromatic errors
        deferred_expressions : bool
            whether to use deferred expressions while importing madx sequence into xsuite
        add_aperture : bool
            whether to call aperture files 
            
        Returns:
        --------
        None
        """
        
        # Load madx instance
        try:
        
            madx = Madx()
            madx.call('{}/LEIR_2021_Pb_ions_thin.seq'.format(sequence_path))
            madx.use(sequence='leir')
        except FileNotFoundError:
            madx = self.load_madx(add_aperture=add_aperture)

        # Load beam parameters from injection energy
        self.m_in_eV, self.gamma_LEIR_inj, self.p_LEIR_extr = self.generate_LEIR_beam()

        # Convert madx sequence to xtrack sequence
        line = xt.Line.from_madx_sequence(madx.sequence['leir'], deferred_expressions=deferred_expressions)
        line.build_tracker()

        # Build reference particle for line
        particle_sample = xp.Particles(
                                        gamma0 = self.gamma_LEIR_inj,
                                        q0 = self.Q_LEIR,
                                        mass0 = self.m_in_eV)

        line.particle_ref = particle_sample
        twiss = line.twiss(method='4d')

        print('\nGenerated LEIR {} beam with gamma = {:.3f}, Qx = {:.3f}, Qy = {:.3f}\n'.format(self.ion_type, 
                                                                                              particle_sample.gamma0[0],
                                                                                              twiss['qx'],
                                                                                              twiss['qy']))
        

        ### SET CAVITY VOLTAGE - with info from Nicolo Biancacci and LSA
        # Ions: we set for nominal cycle V_RF = 3.2 kV and h = 2
        # Two cavities: ER.CRF41 and ER.CRF43 - we use the first of them
        harmonic_nb = 2
        nn = 'er.crf41' # for now test the first of the RF cavities 
        V_RF = 3.2  # kV
        
        # MADX sequence 
        madx.sequence.leir.elements[nn].lag = 0
        madx.sequence.leir.elements[nn].volt = V_RF*1e-3*particle_sample.q0 # different convention between madx and xsuite
        madx.sequence.leir.elements[nn].freq = madx.sequence['leir'].beam.freq0*harmonic_nb
        
        # Xsuite sequence 
        line[nn].lag = 0  # 0 if below transition
        line[nn].voltage =  V_RF*1e3 # In Xsuite for ions, do not multiply by charge as in MADX
        line[nn].frequency = madx.sequence['leir'].beam.freq0*1e6*harmonic_nb

        # Save MADX sequence and xtrack sequence
        if save_madx_seq:                
            madx.command.save(sequence='leir', file='{}/LEIR_2021_Pb_ions.seq'.format(sequence_path), beam=True)

        if save_xsuite_seq:
            with open('{}/LEIR_2021_Pb_ions.json'.format(sequence_path), 'w') as fid:
                json.dump(line.to_dict(), fid, cls=xo.JEncoder)

        if return_xsuite_line:
            return line
        
        
    def plot_twiss_for_LEIR(self, twiss):
    
        """ Plot closed orbit and lattice functions """
        
        plt.close('all')
        
        fig1 = plt.figure(1, figsize=(6.4, 5.5*1.5))
        spbet = plt.subplot(3,1,1)
        spco = plt.subplot(3,1,2, sharex=spbet)
        spdisp = plt.subplot(3,1,3, sharex=spbet)
        
        spbet.plot(twiss.s, twiss.betx)
        spbet.plot(twiss.s, twiss.bety)
        spbet.set_ylabel(r'$\beta_{x,y}$ [m]', fontsize=12)
        
        spco.plot(twiss.s, twiss.x)
        spco.plot(twiss.s, twiss.y)
        spco.set_ylabel(r'(Closed orbit)$_{x,y}$ [m]', fontsize=12)
        
        spdisp.plot(twiss.s, twiss.dx)
        spdisp.plot(twiss.s, twiss.dy)
        spdisp.set_ylabel(r'$D_{x,y}$ [m]', fontsize=12)
        spdisp.set_xlabel('s [m]')
        
        fig1.suptitle(
            r'$q_x$ = ' f'{twiss.qx:.5f}' r' $q_y$ = ' f'{twiss.qy:.5f}' '\n'
            r"$Q'_x$ = " f'{twiss.dqx:.2f}' r" $Q'_y$ = " f'{twiss.dqy:.2f}'
            r' $\gamma_{tr}$ = '  f'{1/np.sqrt(twiss.momentum_compaction_factor):.2f}', fontsize=13
        )
        
        fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
        plt.show()

        
    def _print_leir_seq(self, line):            
        """ Print all elements in line """

        my_dict = line.to_dict()
        d =  my_dict["elements"]
        for key, value in d.items():
            #if value['__class__'] == 'Multipole' and value['_order'] == order:
            print('{}: {}'.format(key, value['__class__']))
