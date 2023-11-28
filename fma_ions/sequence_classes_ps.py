"""
Main module for sequence generator container classes for PS 
"""
from dataclasses import dataclass

import os 
from pathlib import Path

import xobjects as xo
import xtrack as xt
import xpart as xp

from scipy import constants
from cpymad.madx import Madx
import json

ps_madx_macro = Path(__file__).resolve().parent.joinpath('../sequences/ps/PS_match_tunes_and_chroma.madx').absolute()

@dataclass
class PS_sequence_maker:
    """ 
    Data class to generate Xsuite line from PS optics repo, selecting
    - qx0, qy0: horizontal and vertical tunes
    - dq1, dq2: X and Y chroma values
    - Q_LEIR: ion charge state in LEIR
    - Q_PS: ion charge state in PS
    - m_ion: ion mass in atomic units
    - Brho: magnetic rigidity in T*m at injection
    - optics: absolute path to optics repository -> cloned from  https://acc-models.web.cern.ch/acc-models/ps/2022/scenarios/lhc_ion/
    Default chromaticity values originate from here. 
    """
    qx0: float = 6.21
    qy0: float = 6.245
    dq1: float =  -5.26716824
    dq2: float = -7.199251093
    # Default chroma values from optics repo
    
    # Define beam type - default is Pb
    ion_type: str = 'Pb'
    seq_name: str = 'nominal'
    seq_folder: str = 'ps'
    LEIR_Brho  = 4.8 # [Tm] -> Brho at PS injection - same as at LEIR extraction
    Q_LEIR: float = 54.
    Q_PS: float = 54.
    m_ion: float = 207.98
    optics: 'str' = '/home/elwaagaa/cernbox/PhD/Projects/acc-models-ps'

    def generate_PS_beam(self):
        """
        Generate correct injection parameters for PS beam

        Returns:
        -------
        ion rest mass in eV, beam momentum in eV/c at PS injection 
        """
        #print('\nCreating MADX-beam of {}\n'.format(self.ion_type))
        m_in_eV = self.m_ion * constants.physical_constants['atomic mass unit-electron volt relationship'][0]   # 1 Dalton in eV/c^2 -- atomic mass unit
        p_inj_PS = 1e9 * (self.LEIR_Brho * self.Q_LEIR) / 3.3356 # in  [eV/c], if q is number of elementary charges

        return m_in_eV, p_inj_PS


    def generate_xsuite_seq(self, save_madx_seq=False, save_xsuite_seq=True, return_xsuite_line=True):
        """
        Load MADX line, match tunes and chroma, add RF and generate Xsuite line
        """
        os.makedirs(self.seq_folder, exist_ok=True)
        print('\nGenerating sequence for {} with qx = {}, qy = {}\n'.format(self.ion_type, self.qx0, self.qy0))
        
        #### Initiate MADX sequence and call the sequence and optics file ####
        madx = Madx()
        madx.call("{}/_scripts/macros.madx".format(self.optics))
        
        madx.call("{}/ps_mu.seq".format(self.optics))
        madx.call("{}/ps_ss.seq".format(self.optics))
        madx.call("{}/scenarios/lhc_ion/1_flat_bottom/ps_fb_ion.str".format(self.optics))
        madx.call('{}'.format(ps_madx_macro))
        
        # Generate PS beam - use default Pb or make custom beam
        m_in_eV, p_inj_PS = self.generate_PS_beam()
        
        madx.input(" \
                   Beam, particle=ion, mass={}, charge={}, pc = {}, sequence='ps'; \
                   DPP:=BEAM->SIGE*(BEAM->ENERGY/BEAM->PC)^2;  \
                   ".format(m_in_eV/1e9, self.Q_PS, p_inj_PS/1e9))   # convert mass to GeV/c^2
           
        
        # When we perform matching, recall the MADX convention that all chromatic functions are multiplied by relativistic beta factor
        # Thus, when we match for a given chromaticity, need to account for this factor to get correct value in Xsuite and PTC
        beta0 = madx.beam.beta 
        madx.input("qx = {}".format(self.qx0))
        madx.input("qy = {}".format(self.qy0))
        madx.input(f"qpx = -5.26716824/{beta0}")
        madx.input(f"qpy = -7.199251093/{beta0}")
        
        # Flatten line
        madx.use("ps")
        madx.input("seqedit, sequence=PS;")
        madx.input("flatten;")
        madx.input("endedit;")
        madx.use("ps")
        madx.input("select, flag=makethin, slice=5, thick=false;")
        madx.input("makethin, sequence=ps, style=teapot, makedipedge=True;")
        madx.use('ps')
        madx.input("exec, match_tunes_and_chroma(qx, qy, qpx, qpy);")
        
        # Create Xsuite line, check that Twiss command works 
        madx.use(sequence='ps')
        twiss_thin = madx.twiss()  
        
        line = xt.Line.from_madx_sequence(madx.sequence['ps'])
        line.build_tracker()
        #madx_beam = madx.sequence['ps'].beam
        
        self.particle_sample = xp.Particles(
                p0c = p_inj_PS,
                q0 = self.Q_PS,
                mass0 = m_in_eV)
        print('\nGenerated PS {} beam with gamma = {:.3f}, Qx = {:.3f}, Qy = {:.3f}\n'.format(self.ion_type, 
                                                                                              self.particle_sample.gamma0[0],
                                                                                              self.qx0,
                                                                                              self.qy0))
        
        line.particle_ref = self.particle_sample
        
        #### SET CAVITY VOLTAGE - with info from Alexandre Lasheen
        # Ions: 10 MHz cavities: 1.7 MV, h=16
        # In the ps_ss.seq, the 10 MHz cavities are PR_ACC10 - there seems to be 12 of them in the straight sections
        harmonic_nb = 16
        nn = 'pa.c10.11'  # for now test the first of the RF cavities 
        V_RF = 38.0958  # kV

        # MADX sequence 
        madx.sequence.ps.elements[nn].lag = 0
        madx.sequence.ps.elements[nn].volt = V_RF*1e-3*self.particle_sample.q0 # different convention between madx and xsuite
        madx.sequence.ps.elements[nn].freq = madx.sequence['ps'].beam.freq0*harmonic_nb

        # Xsuite sequence 
        line[nn].lag = 0  # 0 if below transition
        line[nn].voltage =  V_RF*1e3 # In Xsuite for ions, do not multiply by charge as in MADX
        line[nn].frequency = madx.sequence['ps'].beam.freq0*1e6*harmonic_nb
        
        # Save MADX sequence
        if save_madx_seq:
            madx.command.save(sequence='ps', file='{}/PS_2022_{}_{}.seq'.format(self.seq_folder, 
                                                                                  self.ion_type, 
                                                                                  self.seq_name), beam=True)  
        # Save Xsuite sequence
        if save_xsuite_seq:
            with open('{}/PS_2022_{}_{}.json'.format(self.seq_folder, self.ion_type, self.seq_name), 'w') as fid:
                json.dump(line.to_dict(), fid, cls=xo.JEncoder)
                
        if return_xsuite_line:
            return line