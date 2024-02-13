"""
Tester script to load proper LEIR sequence
"""
import fma_ions

leir = fma_ions.LEIR_sequence_maker()
line, twiss = leir.load_xsuite_line_and_twiss()

# Get gamma from proper reference particle
beta0 = line.particle_ref.beta0[0]

print('Generated LEIR Xsuite sequence with Qx={:.4f}, Qy={:4f}, dq1={:.4e}, dq2={:.4e}'.format(
   twiss['qx'], twiss['qy'], twiss['dqx'] * beta0, twiss['dqy'] * beta0 ))

# Print sequence elements, and plot Twiss
#leir._print_leir_seq(line)
leir.plot_twiss_for_LEIR(twiss)