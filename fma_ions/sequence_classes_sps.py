"""
Main module for sequence generator container classes for SPS 
"""
from dataclasses import dataclass

import os 

import xobjects as xo
import xtrack as xt
import xpart as xp

from scipy import constants
from cpymad.madx import Madx
import json


@dataclass
class SPS_sequence_maker:
    """ 
    Data class to generate Xsuite line from SPS optics repo, selecting
    - qx0, qy0: horizontal and vertical tunes
    - dq1, dq2: X and Y chroma values 
    - Q_PS: ion charge state in PS
    - Q_SPS: ion charge state in SPS 
    - m_ion: ion mass in atomic units
    - Brho: magnetic rigidity in T*m at injection
    - optics: absolute path to optics repository -> cloned from https://gitlab.cern.ch/acc-models
    """
    qx0: float = 26.30
    qy0: float = 26.25
    dq1: float = -3.460734474533172e-09 
    dq2: float = -3.14426538905229e-09
    # Default SPS PB ION CHROMA VALUES: not displayed on acc-model, extracted from PTC Twiss 
    
    # Define beam type - default is Pb
    ion_type: str = 'Pb'
    seq_name: str = 'nominal'
    seq_folder: str = 'sps'
    B_PS_extr: float = 1.2368 # [T] - magnetic field in PS for Pb ions, from Heiko Damerau
    Q_PS: float = 54.
    Q_SPS: float = 82.
    m_ion: float = 207.98
    optics: 'str' = '/home/elwaagaa/cernbox/PhD/Projects/acc-models-sps'
    

    def generate_SPS_beam(self):
        """
        Generate correct injection parameters for SPS beam
        
        Returns:
        -------
        ion rest mass in eV, beam momentum in eV/c at SPS injection 
        """
        # Calculate Brho at SPS injection
        rho_PS = 70.1206 # [m] - PS bending radius 
        self._Brho_PS_extr = self.B_PS_extr * rho_PS
        
        #print('\nCreating MADX-beam of {}\n'.format(self.ion_type))
        m_in_eV = self.m_ion * constants.physical_constants['atomic mass unit-electron volt relationship'][0]   # 1 Dalton in eV/c^2 -- atomic mass unit
        p_inj_SPS = 1e9 * (self._Brho_PS_extr * self.Q_PS) / 3.3356 # in  [eV/c], if q is number of elementary charges

        return m_in_eV, p_inj_SPS


    def generate_xsuite_seq(self, save_madx_seq=False, save_xsuite_seq=False, return_xsuite_line=True):
        """
        Load MADX line, match tunes and chroma, add RF and generate Xsuite line
        """
        os.makedirs(self.seq_folder, exist_ok=True)
        print('\nGenerating sequence for {} with qx = {}, qy = {}\n'.format(self.ion_type, self.qx0, self.qy0))
        
        #### Initiate MADX sequence and call the sequence and optics file ####
        madx = Madx()
        madx.call("{}/sps.seq".format(self.optics))
        madx.call("{}/strengths/lhc_ion.str".format(self.optics))
        
        # Generate SPS beam - use default Pb or make custom beam
        m_in_eV, p_inj_SPS = self.generate_SPS_beam()
        
        #madx.call("{}/beams/beam_lhc_ion_injection.madx".format(self.optics))
        madx.input(" \
                   Beam, particle=ion, mass={}, charge={}, pc = {}, sequence='sps'; \
                   DPP:=BEAM->SIGE*(BEAM->ENERGY/BEAM->PC)^2;  \
                   ".format(m_in_eV/1e9, self.Q_SPS, p_inj_SPS/1e9))   # convert mass to GeV/c^2
         
        # Flatten line
        madx.use(sequence='sps')
        madx.input("seqedit, sequence=SPS;")
        madx.input("flatten;")
        madx.input("endedit;")
        madx.use("sps")
        madx.input("select, flag=makethin, slice=5, thick=false;")
        madx.input("makethin, sequence=sps, style=teapot, makedipedge=True;")
        
        # Use correct tune and chromaticity matching macros
        madx.call("{}/toolkit/macro.madx".format(self.optics))
        madx.use('sps')
        madx.exec(f"sps_match_tunes({self.qx0}, {self.qy0});")
        madx.exec("sps_define_sext_knobs();")
        madx.exec("sps_set_chroma_weights_q26();")
        madx.input(f"""match;
        global, dq1={self.dq1};
        global, dq2={self.dq2};
        vary, name=qph_setvalue;
        vary, name=qpv_setvalue;
        jacobian, calls=10, tolerance=1e-25;
        endmatch;""")
        
        # Create Xsuite line, check that Twiss command works 
        madx.use(sequence='sps')
        twiss_thin = madx.twiss()  
        
        line = xt.Line.from_madx_sequence(madx.sequence['sps'])
        line.build_tracker()
        #madx_beam = madx.sequence['sps'].beam
        
        self.particle_sample = xp.Particles(
                p0c = p_inj_SPS,
                q0 = self.Q_SPS,
                mass0 = m_in_eV)
        print('\nGenerated {} beam with gamma = {:.3f}\n'.format(self.ion_type, self.particle_sample.gamma0[0]))
        
        line.particle_ref = self.particle_sample
        
        #### SET CAVITY VOLTAGE - with info from Hannes
        # 6x200 MHz cavities: actcse, actcsf, actcsh, actcsi (3 modules), actcsg, actcsj (4 modules)
        # acl 800 MHz cavities
        # acfca crab cavities
        # Ions: all 200 MHz cavities: 1.7 MV, h=4653
        harmonic_nb = 4653
        nn = 'actcse.31632'
        
        # MADX sequence 
        madx.sequence.sps.elements[nn].lag = 0
        madx.sequence.sps.elements[nn].volt = 3.0*self.particle_sample.q0 # different convention between madx and xsuite
        madx.sequence.sps.elements[nn].freq = madx.sequence['sps'].beam.freq0*harmonic_nb
        
        # Xsuite sequence 
        line[nn].lag = 0  # 0 if below transition
        line[nn].voltage =  3.0e6 # In Xsuite for ions, do not multiply by charge as in MADX
        line[nn].frequency = madx.sequence['sps'].beam.freq0*1e6*harmonic_nb
        
        # Save MADX sequence
        if save_madx_seq:
            madx.command.save(sequence='sps', file='{}/SPS_2021_{}_{}.seq'.format(self.seq_folder, 
                                                                                  self.ion_type, 
                                                                                  self.seq_name), beam=True)  
        # Save Xsuite sequence
        if save_xsuite_seq:
            with open('{}/SPS_2021_{}_{}.json'.format(self.seq_folder, self.ion_type, self.seq_name), 'w') as fid:
                json.dump(line.to_dict(), fid, cls=xo.JEncoder)
                
        if return_xsuite_line:
            return line
