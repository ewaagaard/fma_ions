"""
Generate SPS FMA plot - with O ions at different tune
"""
import fma_ions

# Initialize FMA object
fma_sps = fma_ions.FMA(num_turns=120, n_theta=30, n_r=50, output_folder='output_custom_beam')


# Run the quick test FMA analysis
fma_sps.run_custom_beam_SPS(ion_type='O', m_ion=15.99, Q_SPS=8., Q_PS=4., qx=26.30, qy=26.19, Nb=82e8)