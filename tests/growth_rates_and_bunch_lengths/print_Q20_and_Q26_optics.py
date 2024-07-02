"""
Print kinetic and analytical growth rates of Q20 vs Q26 optics
"""
import fma_ions

sps_q20 = fma_ions.SPS_Flat_Bottom_Tracker(num_part=1_000_000, qx0=20.3, qy0=20.25, proton_optics='q20')
sps_q26 = fma_ions.SPS_Flat_Bottom_Tracker(num_part=1_000_000)

print('Q20:')
sps_q20.print_kinetic_and_analytical_growth_rates(distribution_type='binomial')

print('\nQ26:')
sps_q26.print_kinetic_and_analytical_growth_rates(distribution_type='binomial')