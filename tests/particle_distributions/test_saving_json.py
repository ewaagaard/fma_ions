import fma_ions
import matplotlib.pyplot as plt
import os
import json
import xobjects as xo
import xpart as xp

# Track with 10% BB and space charge 
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=100, num_turns=60, turn_print_interval=10)
tbt_dict = sps.track_SPS(which_context='gpu', install_SC_on_line=False, save_full_particle_data=True, full_particle_data_interval=10)

# Save dict and load dict
tbt_dict.to_json('tbt.json')
loaded_data = fma_ions.Full_Records.from_json("tbt.json")