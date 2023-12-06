"""
Simple Dynamic Aperture (DA) test for SPS - with different momentum offsets
"""
import fma_ions

da_sps_0 = fma_ions.Dynamic_Aperture(delta0=0.0)
da_sps_0.run_SPS(case_name='SPS_on_momentum') 

da_sps_1 = fma_ions.Dynamic_Aperture(delta0=1e-3)
da_sps_1.run_SPS(case_name='SPS_delta_0dot001') 

da_sps_2 = fma_ions.Dynamic_Aperture(delta0=-1e-3)
da_sps_2.run_SPS(case_name='SPS_delta_minus_0dot001') 