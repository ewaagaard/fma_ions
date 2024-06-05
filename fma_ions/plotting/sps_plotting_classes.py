"""
Container for all plotting classes relevant to treating SPS data
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class SPS_Plotting:
    
    def load_tbt_data(self, output_folder=None) -> Records:
        """
        Loads numpy data if tracking has already been made
        """
        folder_path = '{}/'.format(output_folder) if output_folder is not None else ''

        # Read the parquet file
        tbt = pd.read_parquet('{}tbt.parquet'.format(folder_path))
        return tbt


    def load_full_records_json(self, output_folder=None, return_dictionary=False):
        """
        Loads json file with full particle data from tracking
        """
        folder_path = '{}/'.format(output_folder) if output_folder is not None else ''

        # Read the json file, return either instanced class or dictionary (if WS profile data is there)
        if return_dictionary:
            tbt = Full_Records.dict_from_json("{}tbt.json".format(folder_path))
        else:
            tbt = Full_Records.from_json("{}tbt.json".format(folder_path))

        return tbt


    def plot_tracking_data(self, 
                           tbt : pd.DataFrame, 
                           include_emittance_measurements=False,
                           show_plot=False,
                           x_unit_in_turns=True,
                           plot_bunch_length_measurements=False):
        """
        Generates emittance plots from turn-by-turn (TBT) data class from simulations,
        compare with emittance measurements (default 2023-10-16) if desired.
        
        Parameters:
        tbt : pd.DataFrame
            dataframe containing the TBT data
        include_emittance_measurements : bool
            whether to include measured emittance or not
        show_plot : bool
            whether to run "plt.show()" in addtion
        x_units_in_turns : bool
            if True, x axis units will be turn, otherwise in seconds
        plot_bunch_length_measurements : bool
            whether to include bunch length measurements from SPS wall current monitor from 2016 studies by Hannes and Tomas
        """

        # If bunch length measurements present, need to plot in seconds
        if plot_bunch_length_measurements:
            x_unit_in_turns = False
            
            # Load bunch length data
            sigma_RMS_Gaussian, sigma_RMS_Binomial, sigma_RMS_Gaussian_in_m, sigma_RMS_Binomial_in_m, ctime = self.load_bunch_length_data()

        # Convert measured emittances to turns if this unit is used, otherwise keep seconds
        if x_unit_in_turns:
            turns = np.arange(len(tbt.exn), dtype=int)             
            time_units = turns
            print('Set time units to turns')
        else:
            if 'Seconds' in tbt.columns:
                time_units = tbt['Seconds']
            elif 'full_data_turn_ind' in tbt.columns:
                sps = SPS_sequence_maker()
                _, twiss = sps.load_xsuite_line_and_twiss()
                turns_per_sec = 1 / twiss.T_rev0
                tbt['Seconds'] = tbt['full_data_turn_ind'] /  turns_per_sec
                time_units = tbt['Seconds'].copy()
                print('Set time units to seconds')
                time_units
            else:
                sps = SPS_sequence_maker()
                _, twiss = sps.load_xsuite_line_and_twiss()
                turns_per_sec = 1 / twiss.T_rev0
                seconds = len(tbt.exn) / turns_per_sec # number of seconds we are running for
                tbt['Seconds'] = np.linspace(0.0, seconds, num=int(len(tbt.exn)))
                time_units = tbt['Seconds'].copy()
                print('Set time units to seconds')

        # Load emittance measurements
        if include_emittance_measurements:
            if x_unit_in_turns:
                sps = SPS_sequence_maker()
                _, twiss = sps.load_xsuite_line_and_twiss()
                turns_per_sec = 1 / twiss.T_rev0
            
            full_data = self.load_emittance_data()
            time_units_x = (turns_per_sec * full_data['Ctime_X']) if x_unit_in_turns else full_data['Ctime_X']
            time_units_y = (turns_per_sec * full_data['Ctime_Y']) if x_unit_in_turns else full_data['Ctime_Y']
            
        # Emittances and bunch intensity 
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (14,5))

        ax1.plot(time_units, tbt.exn * 1e6, alpha=0.7, lw=1.5, label='Simulated')
        ax2.plot(time_units, tbt.eyn * 1e6, alpha=0.7, c='orange', lw=1.5, label='Simulated')
        ax3.plot(time_units, tbt.Nb, alpha=0.7, lw=1.5, c='r', label='Bunch intensity')

        if include_emittance_measurements:
            ax1.errorbar(time_units_x, 1e6 * np.array(full_data['N_avg_emitX']), yerr=1e6 * full_data['N_emitX_error'], 
                       color='blue', fmt="o", label="Measured")
            ax2.errorbar(time_units_y, 1e6 * np.array(full_data['N_avg_emitY']), yerr=1e6 * full_data['N_emitY_error'], 
                       color='darkorange', fmt="o", label="Measured")
            
        # Find min and max emittance values - set window limits 
        all_emit = np.concatenate((tbt.exn, tbt.eyn))
        if include_emittance_measurements:
            all_emit = np.concatenate((all_emit, np.array(full_data['N_avg_emitX']), np.array(full_data['N_avg_emitY'])))
        min_emit = 1e6 * np.min(all_emit)
        max_emit = 1e6 * np.max(all_emit)

        ax1.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        ax2.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        ax3.set_xlabel('Turns' if   x_unit_in_turns else 'Time [s]')
        ax1.set_ylabel(r'$\varepsilon_{x}^{n}$ [$\mu$m]')
        ax2.set_ylabel(r'$\varepsilon_{y}^{n}$ [$\mu$m]')
        ax3.set_ylabel(r'$N_{b}$')
        ax3.set_xlabel('Turns')
        ax1.legend()
        ax2.legend()
        ax1.set_ylim(min_emit-0.08, max_emit+0.1)
        ax2.set_ylim(min_emit-0.08, max_emit+0.1)
        f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        # Sigma_delta and bunch length
        f2, ax12 = plt.subplots(1, 1, figsize = (8,6))
        ax12.plot(time_units, tbt.sigma_delta * 1e3, alpha=0.7, lw=1.5, label='$\sigma_{\delta}$')
        f2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        ax12.set_ylabel(r'$\sigma_{\delta}$')
        ax12.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        
        f3, ax22 = plt.subplots(1, 1, figsize = (8,6))
        ax22.plot(time_units, tbt.bunch_length, alpha=0.7, lw=1.5, label='Simulated')
        if plot_bunch_length_measurements:
            ax22.plot(ctime, sigma_RMS_Gaussian_in_m, label='Measured RMS Gaussian')
            ax22.plot(ctime, sigma_RMS_Binomial_in_m, label='Measured RMS Binomial')
        ax22.set_ylabel(r'$\sigma_{z}$ [m]')
        ax22.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        ax22.legend()
        f3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        # Save figures
        os.makedirs(self.output_folder, exist_ok=True)
        f.savefig('{}/epsilon_Nb.png'.format(self.output_folder), dpi=250)
        f2.savefig('{}/sigma_delta.png'.format(self.output_folder), dpi=250)
        f3.savefig('{}/sigma.png'.format(self.output_folder), dpi=250)

        if show_plot:
            plt.show()
        plt.close()


    def plot_WS_profile_monitor_data(self, 
                                     output_folder=None,
                                     index_to_plot=None
                                     ):
        """
        Use WS beam profile monitor data from tracking to plot transverse beam profiles
        
        Parameters:
        -----------
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        index_to_plot : np.ndarray
            which profiles in time to plot. If None, then automatically plot first and last
        """
        tbt_dict = self.load_full_records_json(output_folder=output_folder, return_dictionary=True)

        # If index not provided, select first and last
        if index_to_plot is None:
            index_to_plot = [0, -1]

        # Plot profile of particles
        fig, ax = plt.subplots(1, 1, figsize = (8, 6))
        for i in index_to_plot:
            ax.plot(tbt_dict['monitorH_x_grid'], tbt_dict['monitorH_x_intensity'][i], 
                    label='Turn {}'.format(tbt_dict['full_data_turn_ind'][i]))
        ax.set_xlabel('x [m]')
        ax.set_ylabel('Counts')
        ax.legend()
        plt.tight_layout()

        # Plot profile of particles
        fig2, ax2 = plt.subplots(1, 1, figsize = (8, 6))
        for i in index_to_plot:
            ax2.plot(tbt_dict['monitorV_y_grid'], tbt_dict['monitorV_y_intensity'][i], 
                     label='Turn {}'.format(tbt_dict['full_data_turn_ind'][i]))
        ax2.set_ylabel('Counts')
        ax2.set_xlabel('y [m]')
        ax2.legend()
        plt.tight_layout()
        plt.show()


    def load_tbt_data_and_plot(self, include_emittance_measurements=False, x_unit_in_turns=True, show_plot=False, output_folder=None,
                               plot_bunch_length_measurements=False):
        """Load already tracked data and plot"""
        try:
            tbt = self.load_tbt_data(output_folder=output_folder)
            self.plot_tracking_data(tbt, 
                                    include_emittance_measurements=include_emittance_measurements,
                                    x_unit_in_turns=x_unit_in_turns,
                                    show_plot=show_plot,
                                    plot_bunch_length_measurements=plot_bunch_length_measurements)
        except FileNotFoundError:
            raise FileNotFoundError('Tracking data does not exist - set correct path or generate the data!')
        

@dataclass
class SPS_Plot_Phase_Space:

    def plot_normalized_phase_space_from_tbt(self, 
                                             output_folder=None, 
                                             include_density_map=True, 
                                             use_only_particles_killed_last=False,
                                             plot_min_aperture=True,
                                             min_aperture=0.025):
        """
        Generate normalized phase space in X and Y to follow particle distribution
        
        Parameters:
        -----------
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        include_density_map : bool
            whether to add color gradient of how tightly packed particles are
        use_only_particles_killed_last : bool
            whether to use the 'kill' index only based on particles killed in the last tracking run
        plot_min_aperture : bool
            whether to include line with minimum X and Y aperture along machine
        min_aperture : float
            default minimum aperture in X and Y (TIDP is collimator limiting y-plane at s=463m)
        """
        tbt_dict = self.load_full_records_json(output_folder=output_folder)

        # Output directory
        os.makedirs('output_plots', exist_ok=True)
        
        # Final dead and alive indices
        alive_ind_final = tbt_dict.state[:, -1] > 0
        dead_ind_final = tbt_dict.state[:, -1] < 1
        dead_ind_lost_in_last_round =  (tbt_dict.state[:, -2] > 0) & (tbt_dict.state[:, -1] < 1)  # particles alive in last tracking round but finally dead
        
        if use_only_particles_killed_last:
            dead_ind_final = dead_ind_lost_in_last_round
            alive_ind_final = np.invert(dead_ind_final)
            extra_ind = '_killed_in_last_round'
        else:
            extra_ind = ''

        # Convert to normalized phase space
        sps = SPS_sequence_maker()
        line, twiss = sps.load_xsuite_line_and_twiss(add_aperture=True)
        
        # Check minimum aperture and plot
        if plot_min_aperture:
            line = sps.remove_aperture_below_threshold(line, min_aperture)
            x_ap, y_ap, a = sps.print_smallest_aperture(line)
            ind_x, ind_y = np.argmin(x_ap), np.argmin(y_ap)
            
            # Find beta functions at these points
            df = twiss.to_pandas()
            betx_min_ap = df.iloc[np.abs(df['s'] - a.iloc[ind_x].s).argmin()].betx
            bety_min_ap = df.iloc[np.abs(df['s'] - a.iloc[ind_y].s).argmin()].bety
            
            # Min aperture - convert to normalized coord
            min_aperture_norm = np.array([x_ap[ind_x] / np.sqrt(betx_min_ap), y_ap[ind_y] / np.sqrt(bety_min_ap)])
        
        X = tbt_dict.x / np.sqrt(twiss['betx'][0]) 
        PX = twiss['alfx'][0] / np.sqrt(twiss['betx'][0]) * tbt_dict.x + np.sqrt(twiss['betx'][0]) * tbt_dict.px
        Y = tbt_dict.y / np.sqrt(twiss['bety'][0]) 
        PY = twiss['alfy'][0] / np.sqrt(twiss['bety'][0]) * tbt_dict.y + np.sqrt(twiss['bety'][0]) * tbt_dict.py
        
        planes = ['X', 'Y']
        Us = [X, Y]
        PUs = [PX, PY]
        
        # Iterate over X and Y
        for i, U in enumerate(Us):
            PU = PUs[i]
            
            ### First plot first and last turn of normalized phase space
            # Generate histograms in all planes to inspect distribution
            bin_heights, bin_borders = np.histogram(U[:, 0], bins=60)
            bin_widths = np.diff(bin_borders)
            bin_centers = bin_borders[:-1] + bin_widths / 2
            #bin_heights = bin_heights/np.max(bin_heights) # normalize bin heights
            
            # Only plot final alive particles
            bin_heights2, bin_borders2 = np.histogram(U[alive_ind_final, -1], bins=60)
            bin_widths2 = np.diff(bin_borders2)
            bin_centers2 = bin_borders2[:-1] + bin_widths2 / 2
            #bin_heights2 = bin_heights2/np.max(bin_heights2) # normalize bin heights
            
            # Plot alive particles sorted by density
            if include_density_map:
                # First turn
                x, y = U[alive_ind_final, 0], PU[alive_ind_final, 0]
                xy = np.vstack([x,y]) # Calculate the point density
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()  # Sort the points by density, so that the densest points are plotted last
                x, y, z = x[idx], y[idx], z[idx]
                
                # Last turn
                x2, y2 = U[alive_ind_final, -1], PU[alive_ind_final, -1]
                xy2 = np.vstack([x2, y2]) # Calculate the point density
                z2 = gaussian_kde(xy2)(xy2)
                idx2 = z2.argsort()  # Sort the points by density, so that the densest points are plotted last
                x2, y2, z2 = x2[idx2], y2[idx2], z2[idx2]
    
            # Plot longitudinal phase space, initial and final state
            fig, ax = plt.subplots(3, 1, figsize = (10, 12), sharex=True)
            
            # Plot initial particles
            if include_density_map:
                ax[0].scatter(x, y, c=z, cmap='cool', s=2, label='Alive')
            else:   
                ax[0].plot(U[alive_ind_final, 0], PU[alive_ind_final, 0], '.', 
                    color='blue', markersize=3.6, label='Alive')
            ax[0].plot(U[dead_ind_final, 0], PU[dead_ind_final, 0], '.', 
                    color='darkred', markersize=3.6, label='Finally dead')
            if plot_min_aperture:
                ax[0].axvline(x=min_aperture_norm[i], ls='-', color='red', alpha=0.7, label='Min. aperture')
                ax[0].axvline(x=-min_aperture_norm[i], ls='-', color='red', alpha=0.7, label=None)
    
            # Plot final particles
            if include_density_map:
                ax[1].scatter(x2, y2, c=z2, cmap='cool', s=2, label='Alive')
            else:   
                ax[1].plot(U[alive_ind_final, -1], PU[alive_ind_final, -1], '.', 
                    color='blue', markersize=3.6, label='Alive')
            ax[1].plot(U[dead_ind_final, -1], PU[dead_ind_final, -1], '.', 
                    color='darkred', markersize=3.6, label='Finally dead')
            if plot_min_aperture:
                ax[1].axvline(x=min_aperture_norm[i], ls='-', color='red', alpha=0.7, label='Min. aperture')
                ax[1].axvline(x=-min_aperture_norm[i], ls='-', color='red', alpha=0.7, label=None)
            ax[1].legend(loc='upper right', fontsize=13)
            
            # Plot initial and final particle distribution
            ax[2].bar(bin_centers, bin_heights, width=bin_widths, alpha=1.0, color='darkturquoise', label='Initial')
            ax[2].bar(bin_centers2, bin_heights2, width=bin_widths2, alpha=0.5, color='lime', label='Final (alive)')
            ax[2].legend(loc='upper right', fontsize=13)
            
            # Adjust axis limits and plot turn
            x_lim = np.max(min_aperture_norm) + 0.001
            ax[0].set_ylim(-x_lim, x_lim)
            ax[0].set_xlim(-x_lim, x_lim)
            ax[1].set_ylim(-x_lim, x_lim)

            
            ax[0].text(0.02, 0.91, 'Turn {}'.format(tbt_dict.full_data_turn_ind[0]+1), fontsize=15, transform=ax[0].transAxes)
            ax[1].text(0.02, 0.91, 'Turn {}'.format(tbt_dict.full_data_turn_ind[-1]+1), fontsize=15, transform=ax[1].transAxes)
                
            ax[2].set_xlabel(r'${}$'.format(planes[i]))
            ax[2].set_ylabel('Counts')
            ax[0].set_ylabel('$P{}$'.format(planes[i]))
            ax[1].set_ylabel('$P{}$'.format(planes[i]))
            plt.tight_layout()
            fig.savefig('output_plots/SPS_Pb_{}_phase_space.png'.format(planes[i]), dpi=250)
            plt.close()

    def plot_longitudinal_phase_space_trajectories(self, 
                                                   output_folder=None, 
                                                   include_sps_separatrix=False,
                                                   xlim=0.85,
                                                   ylim=1.4,
                                                   extra_plt_str='',
                                                   scale_factor_Qs=None,
                                                   plot_zeta_delta_in_phase_space=True):
        """
        Plot color-coded trajectories in longitudinal phase space based on turns
        
        Parameters:
        -----------
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        include_sps_separatrix : bool
            whether to plot line of SPS RF seperatrix
        xlim : float
            if not None, boundary in horizontal plane to include
        ylim : float
            if not None, boundary in vertical plane to include
        extra_plt_str : str
            extra name to add to plot
        scale_factor_Qs : float
            if not None, factor by which we scale Qs (V_RF, h) and divide sigma_z and Nb for similar space charge effects'
        plot_zeta_delta_in_phase_space : bool
            whether to plot delta over zeta in phase space. if False, plot zeta over turns
        """
        tbt_dict = self.load_full_records_json(output_folder=output_folder)

        # Output directory
        os.makedirs('output_plots', exist_ok=True)
        
        # Get SPS zeta separatrix
        if include_sps_separatrix:
            sps = SPS_sequence_maker()
            sps_line, twiss = sps.load_xsuite_line_and_twiss()
            
            # Adjust separatrix if we scale synchrotron tune
            if scale_factor_Qs is not None:
                sps_line, _, _ = sps.change_synchrotron_tune_by_factor(scale_factor_Qs, sps_line)
            
            _, zeta_separatrix, delta_separatrix = generate_binomial_distribution_from_PS_extr(num_particles=50,
                                                                             nemitt_x= BeamParameters_SPS.exn, nemitt_y=BeamParameters_SPS.eyn,
                                                                             sigma_z=BeamParameters_SPS.sigma_z, total_intensity_particles=BeamParameters_SPS.Nb,
                                                                             line=sps_line, return_separatrix_coord=True)
            
        # Create a color map based on number of turns
        num_turns = len(tbt_dict.x[0])
        num_particles = len(tbt_dict.x)
        colors = cm.viridis(np.linspace(0, 1, num_turns))    
    
        # plot longitudinal phase space trajectories of all particles
        fig, ax = plt.subplots(1, 1, figsize = (8, 4.5))

        if plot_zeta_delta_in_phase_space:
            # Plot particles in longitudinal phase space, or zeta of penultimate particle over turns
            for i in range(num_particles):
                print(f'Plotting particle {i+1}')
                ax.scatter(tbt_dict.zeta[i, :], tbt_dict.delta[i, :] * 1e3, c=range(num_turns), marker='.')
            if include_sps_separatrix:
                ax.plot(zeta_separatrix, delta_separatrix * 1e3, '-', color='red', alpha=0.7, label='SPS RF separatrix')
                ax.plot(zeta_separatrix, -delta_separatrix * 1e3, '-', color='red', alpha=0.7, label=None)
            if ylim is not None:
                ax.set_ylim(-ylim, ylim)
            if xlim is not None:
                ax.set_xlim(-xlim, xlim)
            ax.set_xlabel(r'$\zeta$ [m]')
            ax.set_ylabel(r'$\delta$ [1e-3]')

            # Adding color bar for the number of turns
            cbar = plt.colorbar(ax.collections[0], ax=ax)
            cbar.set_label('Number of Turns')
            plt.tight_layout()
            fig.savefig('output_plots/SPS_Pb_longitudinal_trajectories{}.png'.format(extra_plt_str), dpi=250)

        else:
            i = num_particles - 2 # select penultimate particle
            turns = np.arange(num_turns)
            ax.plot(turns, tbt_dict.zeta[i, :] / tbt_dict.zeta[i, 0], alpha=0.7)
            ax.set_xlabel('Turns')
            ax.set_ylabel(r'$\zeta$ / $\zeta_{0}$')
            plt.tight_layout()
            fig.savefig('output_plots/zeta_over_turns{}.png'.format(extra_plt_str), dpi=250)


    def plot_multiple_zeta(self, output_str_array, string_array, xlim=None, ylim=1.2):
        """
        Plot zetas from multiple runs
        
        Parameters:
        ----------
        output_str_array : [outfolder, outfolder, ...]
            List containing string for outfolder tbt data
        string_array : [str1, str2, ...]
            List containing strings to explain the respective tbt data objects (which parameters were used)     
        xlim : float
            upper x limit in turns   
        ylim : float
            upper x limit in turns   
        """
        # Initialize figure, and plot all results
        fig, ax = plt.subplots(1, 1, figsize = (8, 4.5))

        tbt_array = []
        for j, output_folder in reversed(list(enumerate(output_str_array))):
            print(f'Plotting case {j+1}')
            self.output_folder = output_folder
            tbt = self.load_full_records_json(output_folder)
            tbt_array.append(tbt)

            # select penultimate particle
            num_particles = len(tbt.x)
            i = num_particles - 2 

            turns = np.arange(len(tbt.zeta[0]))
            ax.plot(turns, tbt.zeta[i, :] / tbt.zeta[i, 0], alpha=0.8, label=string_array[j])
        
        ax.set_xlabel('Turns')
        ax.set_ylabel(r'$\zeta$ / $\zeta_{0}$')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_ylim(0.97, ylim)
        if xlim is not None:
            ax.set_xlim(-10, xlim)
        plt.tight_layout()
        fig.savefig('multiple_zetas_over_turns.png', dpi=250)


    def plot_longitudinal_phase_space_all_slices_from_tbt(self, 
                                                          output_folder=None, 
                                                          include_sps_separatrix=True,
                                                          include_density_map=True, 
                                                          use_only_particles_killed_last=False):
        """
        Generate longitudinal phase space plots for all turns where they have been recorded
        
        Parameters:
        -----------
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        include_sps_separatrix : bool
            whether to plot line of SPS RF seperatrix
        include_density_map : bool
            whether to add color gradient of how tightly packed particles are
        use_only_particles_killed_last : bool
            whether to use the 'kill' index only based on particles killed in the last tracking run
        """
        tbt_dict = self.load_full_records_json(output_folder=output_folder)

        # Output directory
        os.makedirs('output_plots', exist_ok=True)
        
        # Get SPS zeta separatrix
        if include_sps_separatrix:
            sps = SPS_sequence_maker()
            sps_line, twiss = sps.load_xsuite_line_and_twiss()
            _, zeta_separatrix, delta_separatrix = generate_binomial_distribution_from_PS_extr(num_particles=50,
                                                                             nemitt_x= BeamParameters_SPS.exn, nemitt_y=BeamParameters_SPS.eyn,
                                                                             sigma_z=BeamParameters_SPS.sigma_z, total_intensity_particles=BeamParameters_SPS.Nb,
                                                                             line=sps_line, return_separatrix_coord=True)
        # Final dead and alive indices
        alive_ind_final = tbt_dict.state[:, -1] > 0
        dead_ind_final = tbt_dict.state[:, -1] < 1
        dead_ind_lost_in_last_round =  (tbt_dict.state[:, -2] > 0) & (tbt_dict.state[:, -1] < 1)  # particles alive in last tracking round but finally dead
        
        if use_only_particles_killed_last:
            dead_ind_final = dead_ind_lost_in_last_round
            alive_ind_final = np.invert(dead_ind_final)
            extra_ind = '_killed_in_last_round'
        else:
            extra_ind = ''
        
        # Iterate over all turns that were recorded
        for i in range(len(tbt_dict.full_data_turn_ind)):
    
            print('Plotting data from turn {}'.format(tbt_dict.full_data_turn_ind[i]))
            # Plot longitudinal phase space, initial and final state
            plt.close()
            fig, ax = plt.subplots(1, 1, figsize = (8, 4.5))
            
            # Plot alive particles sorted by density
            if include_density_map:
                # First turn
                x, y = tbt_dict.zeta[alive_ind_final, i], tbt_dict.delta[alive_ind_final, i]*1000
                xy = np.vstack([x,y]) # Calculate the point density
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()  # Sort the points by density, so that the densest points are plotted last
                x, y, z = x[idx], y[idx], z[idx]
                
            # Plot initial particles
            if include_density_map:
                ax.scatter(x, y, c=z, cmap='cool', s=2, label='Alive' if not use_only_particles_killed_last else 'Not killed in last turns')
            else:   
                ax.plot(tbt_dict.zeta[alive_ind_final, i], tbt_dict.delta[alive_ind_final, i]*1000, '.', 
                    color='blue', markersize=3.6, label='Alive' if not use_only_particles_killed_last else 'Not killed in last turns')
            ax.plot(tbt_dict.zeta[dead_ind_final, i], tbt_dict.delta[dead_ind_final, i]*1000, '.', 
                    color='darkred', markersize=3.6, label='Finally dead' if not use_only_particles_killed_last else 'Killed in last turns')
            if include_sps_separatrix:
                ax.plot(zeta_separatrix, delta_separatrix * 1e3, '-', color='red', alpha=0.7, label='SPS RF separatrix')
                ax.plot(zeta_separatrix, -delta_separatrix * 1e3, '-', color='red', alpha=0.7, label=None)
            ax.set_ylim(-1.4, 1.4)
            ax.set_xlim(-0.85, 0.85)
            ax.text(0.02, 0.91, 'Turn {}'.format(tbt_dict.full_data_turn_ind[i]), fontsize=15, transform=ax.transAxes)
            
            ax.legend(loc='upper right', fontsize=11)
            ax.set_xlabel(r'$\zeta$ [m]')
            ax.set_ylabel(r'$\delta$ [1e-3]')
            plt.tight_layout()
            fig.savefig('output_plots/SPS_Pb_longitudinal{}_turn_{}.png'.format(extra_ind, int(tbt_dict.full_data_turn_ind[i])), dpi=250)
        
        
    def plot_longitudinal_phase_space_tbt_from_index(self,
                                                     output_folder=None, 
                                                     random_index_nr = 10, 
                                                     include_sps_separatrix=True):
        """
        Plot killed particles up to index nr in longitudinal phase space
        
        Parameters:
        -----------
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        random_index_nr : int
            among killed particles, select first particles up to this index 
        include_sps_separatrix : bool
            whether to plot line of SPS RF seperatrix
        """    
        tbt_dict = self.load_full_records_json(output_folder=output_folder)

        # Output directory
        os.makedirs('output_plots', exist_ok=True)
        
        # Get SPS zeta separatrix
        if include_sps_separatrix:
            sps = SPS_sequence_maker()
            sps_line, twiss = sps.load_xsuite_line_and_twiss()
            _, zeta_separatrix, delta_separatrix = generate_binomial_distribution_from_PS_extr(num_particles=50,
                                                                             nemitt_x= BeamParameters_SPS.exn, nemitt_y=BeamParameters_SPS.eyn,
                                                                             sigma_z=BeamParameters_SPS.sigma_z, total_intensity_particles=BeamParameters_SPS.Nb,
                                                                             line=sps_line, return_separatrix_coord=True)
        # Final dead and alive indices
        dead_ind_final = tbt_dict.state[:, -1] < 1
        first_random_dead = np.where(dead_ind_final > 0)[0][:random_index_nr]
        
        # Select subset of zeta and delta
        zeta = tbt_dict.zeta[first_random_dead, :]
        delta = tbt_dict.delta[first_random_dead, :]
        colors = cm.winter(np.linspace(0, 1, len(zeta)))
        
        # Iterate over all turns that were recorded
        for i in range(len(tbt_dict.full_data_turn_ind)):
    
            print('Plotting data from turn {}'.format(tbt_dict.full_data_turn_ind[i]))
            # Plot longitudinal phase space, initial and final state
            plt.close()
            fig, ax = plt.subplots(1, 1, figsize = (8, 4.5))
            
            ax.scatter(zeta[:, i], delta[:, i]*1000, c=range(len(zeta)), marker='.')
            if include_sps_separatrix:
                ax.plot(zeta_separatrix, delta_separatrix * 1e3, '-', color='red', alpha=0.7, label='SPS RF separatrix')
                ax.plot(zeta_separatrix, -delta_separatrix * 1e3, '-', color='red', alpha=0.7, label=None)
            ax.set_ylim(-1.4, 1.4)
            ax.set_xlim(-0.85, 0.85)
            ax.text(0.02, 0.91, 'Turn {}'.format(tbt_dict.full_data_turn_ind[i]), fontsize=15, transform=ax.transAxes)
            
            #ax.legend(loc='upper right', fontsize=11)
            ax.set_xlabel(r'$\zeta$ [m]')
            ax.set_ylabel(r'$\delta$ [1e-3]')
            plt.tight_layout()
            fig.savefig('output_plots/SPS_Pb_longitudinal_first{}particles_turn_{}.png'.format(random_index_nr, int(tbt_dict.full_data_turn_ind[i])), dpi=250)
        
        
    def plot_last_and_first_turn_longitudinal_phase_space_from_tbt(self, 
                                                                   output_folder=None, 
                                                                   include_sps_separatrix=False,
                                                                   include_density_map=True):
        """
        Generate longitudinal phase space plots from full particle tracking data
        
        Parameters:
        -----------
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        include_sps_separatrix : bool
            whether to plot line of SPS RF seperatrix
        include_density_map : bool
            whether to add color gradient of how tightly packed particles are
        """
        
        tbt_dict = self.load_full_records_json(output_folder=output_folder)

        # Output directory
        os.makedirs('output_plots', exist_ok=True)
        
        # Get SPS zeta separatrix
        if include_sps_separatrix:
            sps = SPS_sequence_maker()
            sps_line, twiss = sps.load_xsuite_line_and_twiss()
            _, zeta_separatrix, delta_separatrix = generate_binomial_distribution_from_PS_extr(num_particles=50,
                                                                             nemitt_x= BeamParameters_SPS.exn, nemitt_y=BeamParameters_SPS.eyn,
                                                                             sigma_z=BeamParameters_SPS.sigma_z, total_intensity_particles=BeamParameters_SPS.Nb,
                                                                             line=sps_line, return_separatrix_coord=True)
        
        # Final dead and alive indices
        alive_ind_final = tbt_dict.state[:, -1] > 0
        dead_ind_final = tbt_dict.state[:, -1] < 1
        
        # Generate histograms in all planes to inspect distribution
        bin_heights, bin_borders = np.histogram(tbt_dict.zeta[:, 0], bins=60)
        bin_widths = np.diff(bin_borders)
        bin_centers = bin_borders[:-1] + bin_widths / 2
        #bin_heights = bin_heights/np.max(bin_heights) # normalize bin heights
        
        # Only plot final alive particles
        bin_heights2, bin_borders2 = np.histogram(tbt_dict.zeta[alive_ind_final, -1], bins=60)
        bin_widths2 = np.diff(bin_borders2)
        bin_centers2 = bin_borders2[:-1] + bin_widths2 / 2
        #bin_heights2 = bin_heights2/np.max(bin_heights2) # normalize bin heights
        
        # Plot alive particles sorted by density
        if include_density_map:
            # First turn
            x, y = tbt_dict.zeta[alive_ind_final, 0], tbt_dict.delta[alive_ind_final, 0]*1000
            xy = np.vstack([x,y]) # Calculate the point density
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()  # Sort the points by density, so that the densest points are plotted last
            x, y, z = x[idx], y[idx], z[idx]
            
            # Last turn
            x2, y2 = tbt_dict.zeta[alive_ind_final, -1], tbt_dict.delta[alive_ind_final, -1]*1000
            xy2 = np.vstack([x2, y2]) # Calculate the point density
            z2 = gaussian_kde(xy2)(xy2)
            idx2 = z2.argsort()  # Sort the points by density, so that the densest points are plotted last
            x2, y2, z2 = x2[idx2], y2[idx2], z2[idx2]

        # Plot longitudinal phase space, initial and final state
        fig, ax = plt.subplots(3, 1, figsize = (10, 12), sharex=True)
        
        # Plot initial particles
        if include_density_map:
            ax[0].scatter(x, y, c=z, cmap='cool', s=2, label='Alive')
        else:   
            ax[0].plot(tbt_dict.zeta[alive_ind_final, 0], tbt_dict.delta[alive_ind_final, 0]*1000, '.', 
                color='blue', markersize=3.6, label='Alive')
        ax[0].plot(tbt_dict.zeta[dead_ind_final, 0], tbt_dict.delta[dead_ind_final, 0]*1000, '.', 
                color='darkred', markersize=3.6, label='Finally dead')
        if include_sps_separatrix:
            ax[0].plot(zeta_separatrix, delta_separatrix * 1e3, '-', color='red', alpha=0.7, label='SPS RF separatrix')
            ax[0].plot(zeta_separatrix, -delta_separatrix * 1e3, '-', color='red', alpha=0.7, label=None)
        
        # Plot final particles
        if include_density_map:
            ax[1].scatter(x2, y2, c=z2, cmap='cool', s=2, label='Alive')
        else:   
            ax[1].plot(tbt_dict.zeta[alive_ind_final, -1], tbt_dict.delta[alive_ind_final, -1]*1000, '.', 
                color='blue', markersize=3.6, label='Alive')
        ax[1].plot(tbt_dict.zeta[dead_ind_final, -1], tbt_dict.delta[dead_ind_final, -1]*1000, '.', 
                color='darkred', markersize=3.6, label='Finally dead')
        if include_sps_separatrix:
            ax[1].plot(zeta_separatrix, delta_separatrix * 1e3, '-', color='red', alpha=0.7, label='SPS RF separatrix')
            ax[1].plot(zeta_separatrix, -delta_separatrix * 1e3, '-', color='red', alpha=0.7, label=None)
        ax[1].legend(loc='upper right', fontsize=13)
        
        # Plot initial and final particle distribution
        ax[2].bar(bin_centers, bin_heights, width=bin_widths, alpha=1.0, color='darkturquoise', label='Initial')
        ax[2].bar(bin_centers2, bin_heights2, width=bin_widths2, alpha=0.5, color='lime', label='Final (alive)')
        ax[2].legend(loc='upper right', fontsize=13)
        
        # Adjust axis limits and plot turn
        ax[0].set_ylim(-1.4, 1.4)
        ax[0].set_xlim(-0.85, 0.85)
        ax[1].set_ylim(-1.4, 1.4)
        ax[1].set_xlim(-0.85, 0.85)
        ax[2].set_xlim(-0.85, 0.85)
        #ax[2].set_ylim(-0.05, 1.1)
        
        ax[0].text(0.02, 0.91, 'Turn {}'.format(tbt_dict.full_data_turn_ind[0]+1), fontsize=15, transform=ax[0].transAxes)
        ax[1].text(0.02, 0.91, 'Turn {}'.format(tbt_dict.full_data_turn_ind[-1]+1), fontsize=15, transform=ax[1].transAxes)
            
        ax[2].set_xlabel(r'$\zeta$ [m]')
        ax[2].set_ylabel('Counts')
        ax[0].set_ylabel(r'$\delta$ [1e-3]')
        ax[1].set_ylabel(r'$\delta$ [1e-3]')
        plt.tight_layout()
        fig.savefig('output_plots/SPS_Pb_longitudinal.png', dpi=250)
        plt.show()


    def compare_longitudinal_phase_space_vs_data(self, 
                                                 tbt_dict=None,
                                                 output_folder=None,
                                                 also_include_profile_data=True,
                                                 include_final_turn=True,
                                                 num_bins=40,
                                                 final_profile_time_in_s=20.0,
                                                 plot_closest_to_last_profile_instead_of_last_turn=True,
                                                 generate_new_zero_turn_binomial_particle_data_without_pretracking=True,
                                                 also_show_SPS_inj_profile_after_RF_spill=False,
                                                 read_format_is_dict=True):
        """
        Compare measured longitidinal profiles at SPS injection vs generated particles 
        
        Parameters:
        -----------
        tbt_dict : dict
            turn-by-turn dictionary with particle data. Default None will load dictionary from json file
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        also_include_profile_data : bool
            whether to also include wall current monitor data from SPS and BSM data from PS extraction
        include_final_turn : bool
            whether to plot particle data from the final turn
        num_bins : int
            number of bins to include in histogram
        final_profile_time_in_s : float
            time in seconds where the last profile has been recorded
        plot_closest_to_last_profile_instead_of_last_turn : bool
            whether to plot final profile as close to final_profile_time_in_s as possible, otherwise plot last turn
        generate_new_zero_turn_binomial_particle_data_without_pretracking : bool
            if pre-tracking has been done for binomial distribution and particles outside of RF bucket have been removed,
            setting this to True will re-generate a binomial particle distribution with default parameters
        also_show_SPS_inj_profile_after_RF_spill : bool
            if True, show plot after a few milliseconds when SPS RF bucket spill has finished
        read_format_is_dict : bool
            whether to load json file as Full_Records class or as dictionary
        """
        if tbt_dict is None:
            tbt_dict = self.load_full_records_json(output_folder=output_folder, return_dictionary=read_format_is_dict)

            # Read data in correct way
            if read_format_is_dict:
                full_data_turn_ind = np.array(tbt_dict['full_data_turn_ind'])
                state = np.array(tbt_dict['state'])
                zeta = np.array(tbt_dict['zeta'])
            else:
                full_data_turn_ind = tbt_dict.full_data_turn_ind
                state = tbt_dict.state
                zeta = tbt_dict.zeta

        # Output directory
        os.makedirs('output_plots', exist_ok=True)
        
        # Find final index corresponding closest to 20 s (where extraction data is)
        sps = SPS_sequence_maker()
        line, twiss = sps.load_xsuite_line_and_twiss()
        turns_per_sec = 1 / twiss.T_rev0
        full_data_turns_seconds_index = full_data_turn_ind / turns_per_sec # number of seconds we are running for
        closest_index = np.array(np.abs(full_data_turns_seconds_index - final_profile_time_in_s)).argmin()
        
        # Select final index to plot
        ind_final = closest_index if plot_closest_to_last_profile_instead_of_last_turn else -1
        
        # Final dead and alive indices
        alive_ind_final = state[:, ind_final] > 0
        
        # If pre-tracking has been done, decide whether to generate new binomial particle distribution before pre-tracking
        # SHOULD BE FALSE if interested in after the RF spill
        if generate_new_zero_turn_binomial_particle_data_without_pretracking:
            self.num_part = len(zeta[:, 0])
            print('\nGenerating new binomial distribution of {} particles before pre-tracking...'.format(self.num_part))
            context = xo.ContextCpu()
            particles = self.generate_particles(line, context, distribution_type='binomial')
            initial_zeta = zeta
        else:
            print('\nUse particle data from first turn...')
            initial_zeta = zeta[:, 0]
        
        # Generate histograms in all planes to inspect distribution
        bin_heights, bin_borders = np.histogram(initial_zeta, bins=num_bins)
        bin_widths = np.diff(bin_borders)
        bin_centers = bin_borders[:-1] + bin_widths / 2
        ind_max = np.argmax(bin_heights)
        norm_factor = np.max(bin_heights) # np.mean(bin_heights[ind_max-1:ind_max+1]) # normalize of three values around peak
        bin_heights = bin_heights/norm_factor # normalize bin heights
        
        # Only plot final alive particles
        bin_heights2, bin_borders2 = np.histogram(zeta[alive_ind_final, ind_final], bins=num_bins)
        bin_widths2 = np.diff(bin_borders2)
        bin_centers2 = bin_borders2[:-1] + bin_widths2 / 2
        bin_heights2 = bin_heights2/np.max(bin_heights2) # normalize bin heights
        
        # Load data
        zeta_SPS_inj, zeta_SPS_final, zeta_PS_BSM, data_SPS_inj, data_SPS_final, data_PS_BSM = self.load_longitudinal_profile_data()
        if also_show_SPS_inj_profile_after_RF_spill:
            zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill = self.load_longitudinal_profile_after_SPS_injection_RF_spill()

        # Plot longitudinal phase space, initial and final state
        if include_final_turn:
            fig, ax = plt.subplots(2, 1, figsize = (8, 10), sharex=True)
            
            # Plot initial and final particle distribution
            ax[0].bar(bin_centers, bin_heights, width=bin_widths, alpha=0.8, color='darkturquoise', label='Simulated')
            if also_include_profile_data:
                ax[0].plot(zeta_SPS_inj, data_SPS_inj, label='SPS wall current\nmonitor data at inj')
                if also_show_SPS_inj_profile_after_RF_spill:
                    ax[0].plot(zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill, label='SPS wall current\nmonitor data after RF spill')    
                ax[0].plot(zeta_PS_BSM, data_PS_BSM, label='PS BSM data \nat extraction')
            ax[1].bar(bin_centers2, bin_heights2, width=bin_widths2, alpha=0.8, color='lime', label='Final (alive)')
            if also_include_profile_data:
                ax[1].plot(zeta_SPS_final, data_SPS_final, color='darkgreen', label='SPS wall current\nmonitor data (at ~20 s)')
            ax[0].legend(loc='upper right', fontsize=13)
            ax[1].legend(loc='upper right', fontsize=13)
            
            # Adjust axis limits and plot turn
            #ax[0].set_ylim(-1.4, 1.4)
            ax[0].set_xlim(-0.85, 0.85)
            #ax[1].set_ylim(-1.4, 1.4)
            ax[1].set_xlim(-0.85, 0.85)
            
            ax[0].text(0.02, 0.91, 'Turn {}'.format(full_data_turn_ind[0]+1), fontsize=15, transform=ax[0].transAxes)
            ax[1].text(0.02, 0.91, 'Turn {}'.format(full_data_turn_ind[-1]+1), fontsize=15, transform=ax[1].transAxes)
            ax[1].text(0.02, 0.85, 'Time = {:.2f} s'.format(full_data_turns_seconds_index[ind_final]), fontsize=12, transform=ax[1].transAxes)
                
            ax[1].set_xlabel(r'$\zeta$ [m]')
            ax[1].set_ylabel('Counts')
            ax[0].set_ylabel('Normalized count')
            ax[1].set_ylabel('Normalized count')
        else: 
            fig, ax = plt.subplots(1, 1, figsize = (8, 6))

            # Plot initial particle distribution only
            ax.bar(bin_centers, bin_heights, width=bin_widths, alpha=0.8, color='darkturquoise', label='Simulated')
            if also_include_profile_data:
                ax.plot(zeta_SPS_inj, data_SPS_inj, label='SPS wall current monitor data')
                if also_show_SPS_inj_profile_after_RF_spill:
                    ax.plot(zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill, label='SPS wall current\nmonitor data after RF spill')    
                ax.plot(zeta_PS_BSM, data_PS_BSM, label='PS BSM data at extraction')
            ax.legend(loc='upper right', fontsize=13)
            ax.set_xlim(-0.85, 0.85)
            ax.text(0.02, 0.91, 'Turn {}'.format(full_data_turn_ind[0]+1), fontsize=15, transform=ax.transAxes)
                
            ax.set_xlabel(r'$\zeta$ [m]')
            ax.set_ylabel('Counts')
        plt.tight_layout()
        if also_show_SPS_inj_profile_after_RF_spill:
            fig.savefig('output_plots/SPS_Pb_longitudinal_profile_vs_data_after_RF_spill.png', dpi=250)
        else:
            fig.savefig('output_plots/SPS_Pb_longitudinal_profile_vs_data.png', dpi=250)
        plt.show()