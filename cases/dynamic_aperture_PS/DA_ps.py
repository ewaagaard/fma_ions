"""
Simple Dynamic Aperture (DA) test for PS - with different momentum offsets
"""
import fma_ions

da_ps_0 = fma_ions.Dynamic_Aperture(delta0=0.0)
da_ps_0.run_PS(case_name='PS_on_momentum')

da_ps_1 = fma_ions.Dynamic_Aperture(delta0=1e-3)
da_ps_1.run_PS(case_name='PS_delta_0dot001') 

da_ps_2 = fma_ions.Dynamic_Aperture(delta0=-1e-3)
da_ps_2.run_PS(case_name='PS_delta_minus_0dot001') 