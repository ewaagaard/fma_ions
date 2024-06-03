"""
Test loading generated transverse beam profile data and plot profiles
"""
import fma_ions

n_turns = 1000
num_part = 10_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part)
sps.plot_WS_profile_monitor_data()

# Uncomment to load actual tbt dict
#tbt_dict = sps.load_full_records_json(return_dictionary=True)