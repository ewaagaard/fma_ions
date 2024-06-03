"""
Simple test script to launch SPS tracking with WS beam monitor
"""
import fma_ions
output_dir = './'

n_turns = 200
num_part = 1_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part, turn_print_interval=20)
tbt = sps.track_SPS(which_context='cpu', install_SC_on_line=False, full_particle_data_interval=int(5e2), 
                    install_beam_monitors_at_WS_locations=True, save_full_particle_data=True)
tbt.to_json(f'{output_dir}/tbt.json')