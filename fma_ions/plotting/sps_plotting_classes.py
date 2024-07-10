"""
Container for all plotting classes relevant to treating SPS data
"""
from dataclasses import dataclass
from pathlib import Path
import os, json, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import scipy.constants as constants
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
import xobjects as xo

from ..sequences import SPS_sequence_maker, BeamParameters_SPS, BeamParameters_SPS_Oxygen, BeamParameters_SPS_Proton
from ..longitudinal import generate_parabolic_distribution
from ..longitudinal import generate_binomial_distribution_from_PS_extr
from ..helpers_and_functions import Records, Records_Growth_Rates, Full_Records, _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta
from ..helpers_and_functions import Fit_Functions


# Load default emittance measurement data from 2023_10_16
emittance_data_path = Path(__file__).resolve().parent.joinpath('../../data/emittance_data/full_WS_data_SPS_2023_10_16.json').absolute()
Nb_data_path = Path(__file__).resolve().parent.joinpath('../../data/emittance_data/Nb_processed_SPS_2023_10_16.json').absolute()

# Load Pb longitudinal profile measured at PS extraction and SPS injection
longitudinal_data_path = Path(__file__).resolve().parent.joinpath('../../data/longitudinal_profile_data/SPS_inj_longitudinal_data.npy').absolute()
longitudinal_data_path_after_RF_spill = Path(__file__).resolve().parent.joinpath('../../data/longitudinal_profile_data/SPS_inj_longitudinal_data_AFTER_RF_SPILL.npy').absolute()

# Load bunch length data from fitting, choose whether to load data where right tail is cut or not
cut_right_tail_from_fitting_for_bunch_length = True
cut_str = '_cut_right_tail' if cut_right_tail_from_fitting_for_bunch_length else ''
bunch_length_data_path = Path(__file__).resolve().parent.joinpath('../../data/longitudinal_profile_data/SPS_inj_bunch_length_data{}.npy'.format(cut_str)).absolute()

# Load bunch intensity data from 2016 MD
DCBCT_and_WCM_data_path = Path(__file__).resolve().parent.joinpath('../../data/intensity_data/SPS_WCM_and_DCBCT_data.npy')

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 15.5,
        "ytick.labelsize": 15.5,
        "legend.fontsize": 15,
        "figure.titlesize": 20,
    }
)

@dataclass
class SPS_Plotting:
    
    def load_records_dict_from_json(self, output_folder=None):
        """
        Loads json file with particle data from tracking
        """
        folder_path = '{}/'.format(output_folder) if output_folder is not None else ''
        print('Loading data from {}tbt.json'.format(folder_path))

        # Read the json file, return either instanced class or dictionary
        tbt_dict = Records.dict_from_json("{}tbt.json".format(folder_path))

        return tbt_dict


    def plot_tracking_data(self, 
                           tbt_dict=None, 
                           output_folder=None,
                           include_emittance_measurements=False,
                           x_unit_in_turns=False,
                           plot_bunch_length_measurements=True,
                           distribution_type='qgaussian',
                           inj_profile_is_after_RF_spill=True,
                           also_plot_sigma_delta=False,
                           also_plot_WCM_Nb_data=True,
                           also_plot_DCBCT_Nb_data=False,
                           adjusting_factor_Nb_for_initial_drop=0.95,
                           plot_emittances_separately=False,
                           also_plot_particle_std_BL=False):
        """
        Generates emittance plots from turn-by-turn (TBT) data class from simulations,
        compare with emittance measurements (default 2023-10-16) if desired.
        
        Parameters:
        tbt_dict : dict
            dictionary containing the TBT data. If None, loads json file.
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        include_emittance_measurements : bool
            whether to include measured emittance or not
        x_units_in_turns : bool
            if True, x axis units will be turn, otherwise in seconds
        plot_bunch_length_measurements : bool
            whether to include bunch length measurements from SPS wall current monitor from 2016 studies by Hannes and Tomas0
        distribution_type : str
            either 'qgaussian', 'gaussian' or 'binomial'
        inj_profile_is_after_RF_spill : bool
            whether SPS injection profile is after the initial spill out of the RF bucket
        also_plot_sigma_delta : bool
            whether also to plot sigma_delta
        also_plot_WCM_Nb_data : bool
            whether to also plot Wall current monitor data
        also_plot_DCBCT_Nb_data : bool
            whether to also plot DCBCT data
        adjusting_factor_Nb_for_initial_drop : float
            factor by which to multiply WCM data (normalized) times the simulated intensity. A factor 1.0 means that simulations
            started without considering initial RF spill, 0.95 means that the beam parameters were adjusted to after the spill
        also_plot_particle_std_BL : bool
            whether to also plot the standard deviation of particle zeta, i.e. discrete bunch length
        """
        os.makedirs('output_plots', exist_ok=True)
        
        if tbt_dict is None:
            tbt_dict = self.load_records_dict_from_json(output_folder=output_folder)

        # If bunch length measurements present, need to plot in seconds
        if plot_bunch_length_measurements:
            x_unit_in_turns = False
            
            # Load bunch length data
            sigma_RMS_Gaussian_in_m, sigma_RMS_Binomial_in_m, sigma_RMS_qGaussian_in_m, q_measured, dq_measured, ctime = self.load_bunch_length_data()

        # Convert measured emittances to turns if this unit is used, otherwise keep seconds
        if x_unit_in_turns:
            time_units = tbt_dict['Turns']
            print('Set time units to turns')
        else:
            time_units = tbt_dict['Seconds']
            print('Set time units to seconds')

        # Load emittance and intensity measurements
        if include_emittance_measurements:
            if x_unit_in_turns:
                sps = SPS_sequence_maker()
                _, twiss = sps.load_xsuite_line_and_twiss()
                turns_per_sec = 1 / twiss.T_rev0
            
            full_data = self.load_emittance_data()
            time_units_x = (turns_per_sec * full_data['Ctime_X']) if x_unit_in_turns else full_data['Ctime_X']
            time_units_y = (turns_per_sec * full_data['Ctime_Y']) if x_unit_in_turns else full_data['Ctime_Y']
            

        # Load BCT and Wall Current Monitor (WCM) data
        with open(DCBCT_and_WCM_data_path, 'rb') as f:
            time_WCM = np.load(f)
            time_BCT = np.load(f)
            Nb_WCM = np.load(f, allow_pickle=True)
            Nb_BCT_normalized = np.load(f, allow_pickle=True)

        time_units_WCM = (turns_per_sec * time_WCM) if x_unit_in_turns else time_WCM
        time_units_DCBCT = (turns_per_sec * time_BCT) if x_unit_in_turns else time_BCT

        # Emittances and bunch intensity 
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (9.5, 3.6))
        ax1.plot(time_units, tbt_dict['exn'] * 1e6, alpha=0.7, c='turquoise', lw=1.5, label='Simulated')
        ax2.plot(time_units, tbt_dict['eyn'] * 1e6, alpha=0.7, c='turquoise', lw=1.5, label='Simulated')
        if include_emittance_measurements:
            ax1.errorbar(time_units_x, 1e6 * np.array(full_data['N_avg_emitX']), yerr=1e6 * full_data['N_emitX_error'], 
                       color='blue', fmt="o", label="Measured")
            ax2.errorbar(time_units_y, 1e6 * np.array(full_data['N_avg_emitY']), yerr=1e6 * full_data['N_emitY_error'], 
                       color='darkorange', fmt="o", label="Measured")
            
        # Plot bunch intensities, also with mme
        ax3.plot(time_units, tbt_dict['Nb'], alpha=0.7, lw=2.2, c='turquoise', label='Simulated')
        if also_plot_DCBCT_Nb_data:
            ax3.plot(time_units_DCBCT, Nb_BCT_normalized, label='DC-BCT', alpha=0.8, color='b')
        if also_plot_WCM_Nb_data:
            ax3.plot(time_units_WCM, Nb_WCM / adjusting_factor_Nb_for_initial_drop * tbt_dict['Nb'][0],  alpha=0.8,
                      label='Measured', color='r')

        # Find min and max emittance values - set window limits 
        all_emit = np.concatenate((tbt_dict['exn'], tbt_dict['eyn']))
        if include_emittance_measurements:
            all_emit = np.concatenate((all_emit, np.array(full_data['N_avg_emitX']), np.array(full_data['N_avg_emitY'])))
        min_emit = 1e6 * np.min(all_emit)
        max_emit = 1e6 * np.max(all_emit)

        ax1.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        ax2.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        ax3.set_xlabel('Turns' if   x_unit_in_turns else 'Time [s]')
        #plt.setp(ax2.get_yticklabels(), visible=False)
        ax1.set_ylabel(r'$\varepsilon_{x}^{n}$ [$\mu$m]')
        ax2.set_ylabel(r'$\varepsilon_{y}^{n}$ [$\mu$m]')
        ax3.set_ylabel(r'Ions per bunch $N_{b}$')
        ax3.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        ax3.legend(fontsize=12.1, loc='upper right')
        ax1.set_ylim(min_emit-0.08, max_emit+0.1)
        ax2.set_ylim(min_emit-0.08, max_emit+0.1)
        f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        f.savefig('output_plots/epsilon_Nb.png', dpi=250)
        
        # Sigma_delta and bunch length
        if also_plot_sigma_delta:
            f2, ax12 = plt.subplots(1, 1, figsize = (8,6))
            ax12.plot(time_units, tbt_dict['sigma_delta'] * 1e3, alpha=0.7, lw=1.5, label='$\sigma_{\delta}$')
            ax12.set_ylabel(r'$\sigma_{\delta}$')
            ax12.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
            f2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            f2.savefig('output_plots/sigma_delta.png', dpi=250)

        ######### Measured bunch length data and q-values #########
        if distribution_type=='gaussian':
            turn_array, time_array, sigmas_gaussian = self.fit_bunch_lengths_to_data(tbt_dict=tbt_dict, distribution=distribution_type)
        else:
            turn_array, time_array, sigmas_q_gaussian, sigmas_binomial, q, q_error, m, m_error = self.fit_bunch_lengths_to_data(tbt_dict=tbt_dict,
                                                                                                distribution=distribution_type)
       
        f3, ax22 = plt.subplots(1, 1, figsize = (8,6))
        # Uncomment if want to plot standard deviation of numerical particle object
        if also_plot_particle_std_BL:
            ax22.plot(time_units, tbt_dict['bunch_length'], color='darkcyan', alpha=0.7, lw=1.5, label='STD($\zeta$) of simulated particles')      
        
        if distribution_type=='gaussian':
            ax22.plot(turn_array if x_unit_in_turns else time_array, sigmas_gaussian, color='cyan', ls='--', alpha=0.95,
                      label='Simulated profiles')
            if plot_bunch_length_measurements:
                ax22.plot(ctime, sigma_RMS_Gaussian_in_m, color='darkorange', label='Measured profiles')

        elif distribution_type=='binomial':
            ax22.plot(turn_array if x_unit_in_turns else time_array, sigmas_binomial, color='cyan', ls='--', alpha=0.95,
                      label='Simulated profiles')
            if plot_bunch_length_measurements:
                ax22.plot(ctime, sigma_RMS_Binomial_in_m, color='darkorange', alpha=0.95, label='Measured profiles')
        elif distribution_type=='qgaussian':
            ax22.plot(turn_array if x_unit_in_turns else time_array, sigmas_q_gaussian, color='cyan', ls='--', alpha=0.95,
                      label='Simulated profiles')
            if plot_bunch_length_measurements:
                ax22.plot(ctime, sigma_RMS_qGaussian_in_m, color='darkorange', alpha=0.95, label='Measured profiles')
                    
        ax22.set_ylabel(r'$\sigma_{{z, RMS}}$ [m] of fitted {}'.format(distribution_type))
        ax22.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        ax22.legend()
        
        if distribution_type != 'gaussian':
            # Insert extra box with fitted m-value of profiles - plot every 10th value
            ax23 = ax22.inset_axes([0.7,0.5,0.25,0.25])
        
            # Select only reasonable q-values (above 0), then plot only every nth interval
            n_interval = 200
            q_ind = q>0
            q = q[q_ind]
            q_error = q_error[q_ind]
            turn_array_q = turn_array[q_ind]
            time_array_q = time_array[q_ind]
    
            ax23.errorbar(turn_array_q[::n_interval] if x_unit_in_turns else time_array_q[::n_interval], q[::n_interval], yerr=q_error[::n_interval], 
                          color='cyan', alpha=0.85, markerfacecolor='cyan', 
                          ls='None', marker='o', ms=5.1, label='Simulated')
            start_ind = 2 if inj_profile_is_after_RF_spill else 0
            if plot_bunch_length_measurements:
                ax23.errorbar(ctime[start_ind::15], q_measured[start_ind::15], yerr=dq_measured[start_ind::15], markerfacecolor='darkorange', color='darkorange', alpha=0.65, ls='None', marker='o', ms=5.1, label='Measured')
            ax23.set_ylabel('Fitted $q$-value', fontsize=13.5) #, color='green')
            #ax23.legend(fontsize=11, loc='upper left')
            
            ax23.tick_params(axis="both", labelsize=12)
            #ax23.tick_params(colors='green', axis='y')
            ax23.set_ylim(min(q)-0.2, max(q)+0.2)
            ax23.set_xlabel('Time [s]', fontsize=13.5)

        
        f3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        f3.savefig('output_plots/sigma_rms_and_qvalues.png', dpi=250)

        plt.show()


    def fit_bunch_lengths_to_data(self, 
                                  tbt_dict=None,
                                  output_folder=None,
                                  distribution='qgaussian',
                                  show_final_profile=False):
        """
        Fit and extract beam parameters from simulated data, for a given
        distribution type.

        Parameters
        ----------
        tbt_dict : dict
            dictionary containing the TBT data. If None, loads json file.
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        distribution : str
            distribution type of beam: 'binomial', 'qgaussian' or 'gaussian'. The default is 'qgaussian'. 
        show_final_profile : bool
            whether to plot final profile fit vs measurements

        Returns
        -------
        turn_array, time_array, sigmas, m : np.ndarray
            arrays containing turns and time, corresponding to fitted RMS bunch length sigma and m value
            for gaussian profiles, m will only contain zeros

        """
       
        if tbt_dict is None:
            tbt_dict = self.load_records_dict_from_json(output_folder=output_folder)

        # Find total number of stacked profiles and turns per profiles
        n_profiles = len(tbt_dict['z_bin_heights'][0]) 
        nturns_per_profile = tbt_dict['nturns_profile_accumulation_interval']
        sigmas = np.zeros(n_profiles)
        sigmas_binomial = np.zeros(n_profiles)
        sigmas_q_gaussian = np.zeros(n_profiles)
        m = np.zeros(n_profiles)
        m_error = np.zeros(n_profiles)
        q_vals = np.zeros(n_profiles)
        q_errors = np.zeros(n_profiles)
        
        # Create time array with
        turns_per_s = tbt_dict['Turns'][-1] / tbt_dict['Seconds'][-1]
        turn_array = np.arange(0, tbt_dict['Turns'][-1], step=nturns_per_profile)
        time_array = turn_array.copy() / turns_per_s

        # Initiate fit function
        fits = Fit_Functions()

        # Try loading already fitted profiles - otherwise fit them!
        try:
            with open('saved_bunch_length_fits.pickle', 'rb') as handle:
                BL_dict = pickle.load(handle)
            print('Loaded dictionary of fitted bunch lengths')
                
            if distribution=='qgaussian' or distribution=='binomial':
                sigmas_q_gaussian, sigmas_binomial = BL_dict['sigmas_q_gaussian'], BL_dict['sigmas_binomial']
                q_vals, q_errors, m, m_error = BL_dict['q_vals'], BL_dict['q_errors'], BL_dict['m'], BL_dict['m_error']
            elif distribution=='gaussian':
                sigmas = BL_dict['sigmas']
            else:
                raise ValueError('Only binomial, q-gaussian or gaussian distributions implemented')
                

        except FileNotFoundError: 
            
            # Fit binomials, q-gaussian or gaussian bunch lengths
            for i in range(n_profiles):
                z_bin_heights_sorted = np.array(sorted(tbt_dict['z_bin_heights'][:, i], reverse=True))
                z_height_max_avg = np.mean(z_bin_heights_sorted[:5]) # take average of top 5 values
                xdata, ydata = tbt_dict['z_bin_centers'], tbt_dict['z_bin_heights'][:, i] / z_height_max_avg
                            
                if distribution=='qgaussian' or distribution=='binomial':
                        # Fit both q-Gaussian and binomial
                        popt_Q, pcov_Q = fits.fit_Q_Gaussian(xdata, ydata)
                        q_vals[i] = popt_Q[1]
                        q_errors[i] = np.sqrt(np.diag(pcov_Q))[1] # error from covarance_matrix
                        sigmas_q_gaussian[i] = fits.get_sigma_RMS_from_qGaussian_fit(popt_Q)
                        print('Profile {}: q-Gaussian fit q={:.3f} +/- {:.2f}, sigma_RMS = {:.3f} m'.format(i, q_vals[i], q_errors[i], 
                                                                                                                  sigmas_q_gaussian[i]))
                        popt_B, pcov_B = fits.fit_Binomial(xdata, ydata)
                        sigmas_binomial[i], sigmas_error = fits.get_sigma_RMS_from_binomial_fit(popt_B, pcov_B)
                        m[i] = popt_B[1]
                        m_error[i] = np.sqrt(np.diag(pcov_B))[1]
                        print('Profile {}: binomial fit m={:.3f} +/- {:.2f}, sigma_RMS = {:.3f} +/- {:.2f}\n'.format(i, m[i], m_error[i], 
                                                                                                                     sigmas_binomial[i], sigmas_error))
                elif distribution=='gaussian':
                    popt_G, pcov_G = fits.fit_Gaussian(xdata, ydata)
                    sigmas[i] = np.abs(popt_G[2])
                    print('Gaussian: sigma_RMS = {:.3f} m'.format(popt_G[2]))
                else:
                    raise ValueError('Only binomial, q-gaussian or gaussian distributions implemented')
                    
            # Create dictionary with fits
            if distribution=='qgaussian' or distribution=='binomial':
                BL_dict = {'sigmas_q_gaussian': sigmas_q_gaussian, 'sigmas_binomial': sigmas_binomial, 
                           'q_vals': q_vals, 'q_errors': q_errors, 'm': m, 'm_error': m_error}
            else:
                BL_dict = {'sigmas': sigmas}
                    
            # Dump saved fits in dictionary, then pickle file
            with open('saved_bunch_length_fits.pickle', 'wb') as handle:
                pickle.dump(BL_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Dictionary with saved fits dumped')               

        #### First final profile vs fit
        if show_final_profile:
            fig0, ax0 = plt.subplots(1, 1, figsize = (8, 6))
            ax0.plot(xdata, ydata, label='Fit')
            if distribution=='binomial':
                ax0.plot(xdata, fits.Q_Gaussian(xdata, *popt_Q), color='green', ls='--', lw=2.8, label='q-Gaussian fit')
                ax0.plot(xdata, fits.Binomial(xdata, *popt_B), color='red', ls=':', lw=2.8, label='Binomial fit')
            elif distribution=='gaussian':
                ax0.plot(xdata, fits.Gaussian(xdata, *popt_G), color='red', ls='--', lw=2.8, label='Gaussian fit')
            ax0.set_xlabel('$\zeta$ [m]')
            ax0.set_ylabel('Normalized counts')
            ax0.legend(loc='upper left', fontsize=14)
            plt.tight_layout()
            plt.show()

        if distribution=='gaussian':
            return turn_array, time_array, sigmas
        else: 
            return turn_array, time_array, sigmas_q_gaussian, sigmas_binomial, q_vals, q_errors, m, m_error





    def plot_multiple_sets_of_tracking_data(self, 
                                            output_str_array, 
                                            string_array, 
                                            compact_mode=False,
                                            include_emittance_measurements=False, 
                                            emittance_limits=None,
                                            plot_bunch_length_measurements=True,
                                            x_unit_in_turns=False,
                                            bbox_to_anchor_position=(0.0, 1.3),
                                            labelsize = 15.8,
                                            ylim=None, 
                                            legend_font_size=11.4,
                                            extra_str='',
                                            also_plot_WCM_Nb_data=True,
                                            adjusting_factor_Nb_for_initial_drop=0.95,
                                            distribution_type='binomial'):
        """
        If multiple runs with turn-by-turn (tbt) data has been made, provide list with Records class objects and list
        of explaining string to generate comparative plots of emittances, bunch intensities, etc

        Parameters:
        ----------
        output_str_array : [outfolder, outfolder, ...]
            List containing string for outfolder tbt data
        string:_array : [str1, str2, ...]
            List containing strings to explain the respective tbt data objects (which parameters were used)
        compact_mode : bool
            whether to slim plot in more compact format 
        include_emittance_measurements : bool
            whether to include measured emittance or not
        plot_bunch_length_measurements : bool
            whether to include bunch length measurements from SPS wall current monitor from 2016 studies by Hannes and Tomas
        x_units_in_turns : bool
            if True, x axis units will be turn, otherwise in seconds
        bbox_to_anchor_position : tuple
            x-y coordinates of relative plot position for legend
        labelsize : int
            labelsize for axes
        ylim : list
            lower and upper bounds for emittance plots, if None (default), automatic limits are set
        legend_font_size : int
            labelsize for legend
        extra_str : str
            for plotting names
        also_plot_WCM_Nb_data : bool
            whether to also plot Wall current monitor data
        adjusting_factor_Nb_for_initial_drop : float
            factor by which to multiply WCM data (normalized) times the simulated intensity. A factor 1.0 means that simulations
            started without considering initial RF spill, 0.95 means that the beam parameters were adjusted to after the spill
        distribution_type : str
            either 'gaussian' or 'binomial'
        """
        os.makedirs('main_plots', exist_ok=True)
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 16,
                "axes.titlesize": 16,
                "axes.labelsize": labelsize,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": legend_font_size,
                "figure.titlesize": 18,
            }
        )

        # Load TBT data 
        tbt_array = []
        for output_folder in output_str_array:
            self.output_folder = output_folder
            tbt_dict = self.load_records_dict_from_json(output_folder=output_folder)
            tbt_array.append(tbt_dict)

        # Convert measured emittances to turns if this unit is used, otherwise keep seconds
        if x_unit_in_turns:         
            sps = SPS_sequence_maker()
            _, twiss = sps.load_xsuite_line_and_twiss()
            turns_per_sec = 1 / twiss.T_rev0
            time_units = tbt_dict['Turns']
            print('Set time units to turns')
        else:
            time_units = tbt_dict['Seconds']
            print('Set time units to seconds')

        # Load BCT and Wall Current Monitor (WCM) data
        with open(DCBCT_and_WCM_data_path, 'rb') as f:
            time_WCM = np.load(f)
            time_BCT = np.load(f)
            Nb_WCM = np.load(f, allow_pickle=True)
            Nb_BCT_normalized = np.load(f, allow_pickle=True)

        time_units_WCM = (turns_per_sec * time_WCM) if x_unit_in_turns else time_WCM

        # Load emittance measurements
        if include_emittance_measurements:
            if x_unit_in_turns:
                sps = SPS_sequence_maker()
                _, twiss = sps.load_xsuite_line_and_twiss()
                turns_per_sec = 1 / twiss.T_rev0
            
            full_data = self.load_emittance_data()
            time_units_x = (turns_per_sec * full_data['Ctime_X']) if x_unit_in_turns else full_data['Ctime_X']
            time_units_y = (turns_per_sec * full_data['Ctime_Y']) if x_unit_in_turns else full_data['Ctime_Y']

            df_Nb = self.load_Nb_data()
            time_Nb = (turns_per_sec * df_Nb['ctime']) if x_unit_in_turns else df_Nb['ctime']

        # Normal, or compact mode
        if compact_mode:
            
            # Old way for IPAC
            #f = plt.figure(figsize = (6, 6))
            #gs = f.add_gridspec(3, hspace=0, height_ratios= [1, 2, 2])
            #(ax3, ax2, ax1) = gs.subplots(sharex=True, sharey=False)

            f, axs = plt.subplots(2, 2, figsize = (7,5), sharex=True)

            # Plot measurements, if desired                
            if include_emittance_measurements:
                axs[1, 0].plot(time_Nb, df_Nb['Nb'], color='blue', marker="o", ms=2.5, alpha=0.7, label="Measured")
            
            # Loop over the tbt records classes 
            for i, tbt_dict in enumerate(tbt_array):
                axs[0, 0].plot(time_units, tbt_dict['exn'] * 1e6, alpha=0.7, lw=1.5, label=string_array[i])
                axs[0, 1].plot(time_units, tbt_dict['eyn'] * 1e6, alpha=0.7, lw=1.5, label=string_array[i])
                axs[1, 0].plot(time_units, tbt_dict['Nb'], alpha=0.7, lw=2.5, label=None)
                
            # Include wire scanner data - subtract ion injection cycle time
            if include_emittance_measurements:
                axs[0, 0].errorbar(time_units_x - self.ion_inj_ctime, 1e6 * np.array(full_data['N_avg_emitX']), yerr=1e6 * full_data['N_emitX_error'], 
                           color='blue', fmt="o", label="Measured")
                axs[0, 1].errorbar(time_units_y - self.ion_inj_ctime, 1e6 * np.array(full_data['N_avg_emitY']), yerr=1e6 * full_data['N_emitY_error'], 
                           color='blue', fmt="o", label="Measured")
                
            axs[0, 0].set_ylabel(r'$\varepsilon_{x}^{n}$ [$\mu$m rad]')
            axs[0, 1].set_ylabel(r'$\varepsilon_{y}^{n}$ [$\mu$m rad]')
            axs[1, 0].set_ylabel(r'$N_{b}$')
            #axs[0, 0].text(0.94, 0.94, 'X', color='darkgreen', fontsize=20, transform=axs[0, 0].transAxes)
            #axs[0, 1].text(0.02, 0.94, 'Y', color='darkgreen', fontsize=20, transform=axs[0, 1].transAxes)
            
            if also_plot_WCM_Nb_data:
                axs[1, 0].plot(time_units_WCM, Nb_WCM / adjusting_factor_Nb_for_initial_drop * tbt_dict['Nb'][0], ls='--', alpha=0.8,
                          label='Wall Current\nMonitor', color='r')
                axs[1, 0].legend(fontsize=legend_font_size, loc='lower left')
            
            # Bunch length
            for i, tbt_dict in enumerate(tbt_array):
                turn_array, time_array, sigmas, sigmas_error, m, m_error = self.fit_bunch_lengths_to_data(tbt_dict=tbt_dict, distribution=distribution_type)
                axs[1, 1].plot(turn_array if x_unit_in_turns else time_array, sigmas, ls='-', 
                          label=string_array[i])
                
                #axs[1, 1].plot(time_units, tbt_dict['bunch_length'],  alpha=0.7, lw=1.5, label=string_array[i])

            # Load binomial bunch length data and plot
            if plot_bunch_length_measurements:
                sigma_RMS_Gaussian_in_m, sigma_RMS_Binomial_in_m, sigma_RMS_qGaussian_in_m, q_vals, q_error, ctime = self.load_bunch_length_data()
                axs[1, 1].plot(ctime, sigma_RMS_Binomial_in_m, color='orangered', alpha=0.95, ls='--', label='Measured\nbinomial')
                
            # Fix labels and only make top visible
            plt.setp(axs[0, 0].get_xticklabels(), visible=False)
            plt.setp(axs[0, 1].get_xticklabels(), visible=False)
            axs[1, 1].set_ylabel(r'Fitted $\sigma_{z, RMS}$ [m]')
            axs[1, 1].legend(fontsize=legend_font_size, loc='upper right')
            axs[1, 0].set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
            axs[1, 1].set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
            
            if ylim is not None:
                axs[0, 0].set_ylim(ylim[0], ylim[1])
                axs[0, 1].set_ylim(ylim[0], ylim[1])
            #axs[0, 1].legend(fontsize=legend_font_size, loc='upper left', bbox_to_anchor=bbox_to_anchor_position)

            #for ax in f.get_axes():
            #    ax.label_outer()
            
            f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            f.savefig('main_plots/result_multiple_trackings_compact{}.png'.format(extra_str), dpi=250)
            plt.show()
            
        else:
            # Emittances and bunch intensity 
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (14,5))
    
            # Loop over the tbt records classes 
            for i, tbt_dict in enumerate(tbt_array):
                ax1.plot(time_units, tbt_dict['exn'] * 1e6, alpha=0.7, lw=1.5, label=string_array[i])
                ax2.plot(time_units, tbt_dict['eyn'] * 1e6, alpha=0.7, lw=1.5, label=string_array[i])
                ax3.plot(time_units, tbt_dict['Nb'], alpha=0.7, lw=1.5, label=string_array[i])
    
            if include_emittance_measurements:
                ax1.errorbar(time_units_x, 1e6 * np.array(full_data['N_avg_emitX']), yerr=1e6 * full_data['N_emitX_error'], 
                           color='blue', fmt="o", label="Measured")
                ax2.errorbar(time_units_y, 1e6 * np.array(full_data['N_avg_emitY']), yerr=1e6 * full_data['N_emitY_error'], 
                           color='darkorange', fmt="o", label="Measured")
    
            if also_plot_WCM_Nb_data:
                ax3.plot(time_units_WCM, Nb_WCM / adjusting_factor_Nb_for_initial_drop * tbt_dict['Nb'][0], ls='--', alpha=0.8,
                          label='Wall Current Monitor', color='r')
            ax1.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
            ax2.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
            ax3.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
            if ylim is not None:
                ax1.set_ylim(ylim[0], ylim[1])
                ax2.set_ylim(ylim[0], ylim[1])
            ax1.set_ylabel(r'$\varepsilon_{x}^{n}$ [$\mu$m]')
            ax2.set_ylabel(r'$\varepsilon_{y}^{n}$ [$\mu$m]')
            ax3.set_ylabel(r'$N_{b}$')
            ax3.legend(fontsize=13, loc='upper right')
            f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            f.savefig('main_plots/result_multiple_trackings{}.png'.format(extra_str), dpi=250)
            
            
            # Bunch length
            f3, ax22 = plt.subplots(1, 1, figsize = (8,6))
            for i, tbt_dict in enumerate(tbt_array):
                ax22.plot(time_units, tbt_dict['bunch_length'],  alpha=0.7, lw=1.5, label=string_array[i])

            if plot_bunch_length_measurements:
                # Load bunch length data
                sigma_RMS_Gaussian_in_m, sigma_RMS_Binomial_in_m, sigma_RMS_qGaussian_in_m, q_vals, q_error, ctime = self.load_bunch_length_data()
                #ax22.plot(ctime, sigma_RMS_Gaussian_in_m, color='royalblue', ls='-.', label='Measured $\sigma$ Gaussian')
                ax22.plot(ctime, sigma_RMS_Binomial_in_m, color='orangered', alpha=0.7, ls='--', label='Measured RMS Binomial')
            ax22.set_ylabel(r'$\sigma_{z}$ [m]')
            ax22.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
            ax22.legend(fontsize=legend_font_size, loc='upper left')
            f3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            f3.savefig('main_plots/sigma_multiple_trackings{}.png'.format(extra_str), dpi=250)
            
            plt.show()


    def plot_multiple_emittance_runs(self,
                                     output_str_array,
                                     string_array,
                                     plot_moving_average=True,
                                     x_unit_in_turns=False,
                                     include_emittance_measurements=False,
                                     extra_str='',
                                     ylim=None
                                     ):


        """
        Combined beta-beat plot of multiple tracking simulations
        
        output_str_array : [outfolder, outfolder, ...]
            List containing string for outfolder tbt data
        string_array : [str1, str2, ...]
            List containing strings to explain the respective tbt data objects (which parameters were used)
        plot_moving_average : bool
            whether to use a savgol filter to plot the moving average
        x_unit_in_turns : bool
            if True, x axis units will be turn, otherwise in seconds
        include_emittance_measurements : bool
            whether to include data from 2023
        extra_str : str
            add to figure name
        ylim : list
            vertical limits to plot
        """
        # Make plot directory and update plot parameters
        os.makedirs('main_plots', exist_ok=True)
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 16,
                "axes.titlesize": 16,
                "axes.labelsize": 16,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 10.4,
                "figure.titlesize": 18,
            }
        )

        # Load TBT data 
        tbt_array = []
        for output_folder in output_str_array:
            self.output_folder = output_folder
            tbt_dict = self.load_records_dict_from_json(output_folder=output_folder)
            tbt_array.append(tbt_dict)

        # Convert measured emittances to turns if this unit is used, otherwise keep seconds
        if x_unit_in_turns:         
            sps = SPS_sequence_maker()
            _, twiss = sps.load_xsuite_line_and_twiss()
            turns_per_sec = 1 / twiss.T_rev0
            time_units = tbt_dict['Turns']
            print('Set time units to turns')
        else:
            time_units = tbt_dict['Seconds']
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

        # Generate figure
        f, axs = plt.subplots(1, 2, figsize = (7.5, 3.9), sharey=True)
        
        # Loop over the tbt records classes 
        for i, tbt_dict in enumerate(tbt_array):
            if plot_moving_average:
                axs[0].plot(time_units, savgol_filter(tbt_dict['exn'], 1000, 2) * 1e6, alpha=0.7, lw=1.5, label=string_array[i])
                axs[1].plot(time_units, savgol_filter(tbt_dict['eyn'], 1000, 2) * 1e6, alpha=0.7, lw=1.5, label=string_array[i]) 
            else:
                axs[0].plot(time_units, tbt_dict['exn'] * 1e6, alpha=0.7, lw=1.5, label=string_array[i])
                axs[1].plot(time_units, tbt_dict['eyn'] * 1e6, alpha=0.7, lw=1.5, label=string_array[i])
    
        # Include wire scanner data - subtract ion injection cycle time
        if include_emittance_measurements:
            axs[0].errorbar(time_units_x - self.ion_inj_ctime, 1e6 * np.array(full_data['N_avg_emitX']), yerr=1e6 * full_data['N_emitX_error'], 
                       color='blue', fmt="o", label="Measured")
            axs[1].errorbar(time_units_y - self.ion_inj_ctime, 1e6 * np.array(full_data['N_avg_emitY']), yerr=1e6 * full_data['N_emitY_error'], 
                       color='blue', fmt="o", label="Measured")
    
        axs[0].set_ylabel(r'$\varepsilon_{x}^{n}$ [$\mu$m rad]')
        axs[1].set_ylabel(r'$\varepsilon_{y}^{n}$ [$\mu$m rad]')

    
        # Fix labels and only make top visible
        plt.setp(axs[1].get_yticklabels(), visible=False)
        axs[0].legend(loc='lower right')
        axs[0].set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        axs[1].set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        axs[1].text(0.04, 0.91, '{}'.format(extra_str), fontsize=15, transform=axs[1].transAxes)

        if ylim is not None:
            axs[0].set_ylim(ylim[0], ylim[1])
            axs[1].set_ylim(ylim[0], ylim[1])
        
        f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        f.savefig('main_plots/emittance_growths_{}.png'.format(extra_str), dpi=250)
        plt.show()


    def plot_WS_profile_monitor_data(self, 
                                     tbt_dict=None,
                                     output_folder=None,
                                     index_to_plot=None
                                     ):
        """
        Use WS beam profile monitor data from tracking to plot transverse beam profiles
        
        Parameters:
        -----------
        tbt_dict : dict
            dictionary containing turn-by-turn data. If None, will load json file
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        index_to_plot : list
            which profiles in time to plot. If None, then automatically plot second and second-last profile
        """
        os.makedirs('output_plots', exist_ok=True)

        if tbt_dict is None:
            tbt_dict = self.load_records_dict_from_json(output_folder=output_folder)

        # If index not provided, select second and second-to-last sets of 100 turns
        if index_to_plot is None:
            index_to_plot = [1, -2]
            
        # Find total number of stacked profiles and turns per profiles
        stack_index = np.arange(len(tbt_dict['z_bin_heights'][0]))    
        nturns_per_profile = tbt_dict['nturns_profile_accumulation_interval']
        
        # Show time stamp if seconds are available
        if 'Seconds' in tbt_dict:
            turns_per_s = tbt_dict['Turns'][-1] / tbt_dict['Seconds'][-1]
            plot_str=  ['At time = {:.2f} s'.format(nturns_per_profile * (1 + stack_index[index_to_plot[0]]) / turns_per_s), 
                        'At time = {:.2f} s'.format(nturns_per_profile * (1 + stack_index[index_to_plot[1]]) / turns_per_s)]
        else:
            plot_str = ['At turn {}'.format(nturns_per_profile * (1 + stack_index[index_to_plot[0]])), 
                        'At turn {}'.format(nturns_per_profile * (1 + stack_index[index_to_plot[1]]))]

        # Plot profile of particles
        fig, ax = plt.subplots(1, 1, figsize = (8, 6))
        for j, i in enumerate(index_to_plot):
            # Normalize bin heights
            x_bin_heights_sorted = np.array(sorted(tbt_dict['monitorH_x_intensity'][i], reverse=True))
            x_height_max_avg = np.mean(x_bin_heights_sorted[:5]) # take average of top 5 values
            ax.plot(tbt_dict['monitorH_x_grid'], tbt_dict['monitorH_x_intensity'][i] / x_height_max_avg, label=plot_str[j])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('Normalized counts')
        ax.legend(loc='upper left', fontsize=14)
        plt.tight_layout()
        fig.savefig('output_plots/SPS_X_Beam_Profile_WS.png', dpi=250)

        # Plot profile of particles
        fig2, ax2 = plt.subplots(1, 1, figsize = (8, 6))
        for j, i in enumerate(index_to_plot):
            # Normalize bin heights
            y_bin_heights_sorted = np.array(sorted(tbt_dict['monitorV_y_intensity'][i], reverse=True))
            y_height_max_avg = np.mean(y_bin_heights_sorted[:10]) # take average of top ten values
            ax2.plot(tbt_dict['monitorV_y_grid'], tbt_dict['monitorV_y_intensity'][i] / y_height_max_avg, label=plot_str[j])
        ax2.set_ylabel('Normalized counts')
        ax2.set_xlabel('y [m]')
        ax2.legend(loc='upper left', fontsize=14)
        plt.tight_layout()
        fig2.savefig('output_plots/SPS_Y_Beam_Profile_WS.png', dpi=250)
        plt.show()


    def plot_longitudinal_monitor_data(self,
                                       tbt_dict=None,
                                       output_folder=None,
                                       index_to_plot=None,
                                       also_compare_with_profile_data=True,
                                       inj_profile_is_after_RF_spill=True
                                       ):
        """
        Use longitudinal data from tracking to plot beam profile of zeta
        
        Parameters:
        -----------
        tbt_dict : dict
            dictionary containing turn-by-turn data. If None, will load json file
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        index_to_plot : list
            which profiles in time to plot. If None, then automatically plot second and second-last profile
        also_compare_with_profile_data : bool
            whether to include profile measurements
        inj_profile_is_after_RF_spill : bool
            whether SPS injection profile is after the initial spill out of the RF bucket
        """
        os.makedirs('output_plots', exist_ok=True)
        
        if tbt_dict is None:
            tbt_dict = self.load_records_dict_from_json(output_folder=output_folder)

        # If index not provided, select second and second-to-last sets of 100 turns
        if index_to_plot is None:
            index_to_plot = [1, -2]
            
        # Find total number of stacked profiles and turns per profiles
        stack_index = np.arange(len(tbt_dict['z_bin_heights'][0]))    
        nturns_per_profile = tbt_dict['nturns_profile_accumulation_interval']
        
        # Show time stamp if seconds are available
        if 'Seconds' in tbt_dict:
            turns_per_s = tbt_dict['Turns'][-1] / tbt_dict['Seconds'][-1]
            plot_str =  ['At time = {:.2f} s'.format(nturns_per_profile * (1 + stack_index[index_to_plot[0]]) / turns_per_s), 
                        'At time = {:.2f} s'.format(nturns_per_profile * (1 + stack_index[index_to_plot[1]]) / turns_per_s)]
        else:
            plot_str = ['At turn {}'.format(nturns_per_profile * (1 + stack_index[index_to_plot[0]])), 
                        'At turn {}'.format(nturns_per_profile * (1 + stack_index[index_to_plot[1]]))]

        #### First plot initial and final simulated profile
        fig0, ax0 = plt.subplots(1, 1, figsize = (8, 6))
        j = 0
        z_heights_avg = []
        for i in index_to_plot:
            # Normalize bin heights
            z_bin_heights_sorted = np.array(sorted(tbt_dict['z_bin_heights'][:, i], reverse=True))
            z_height_max_avg = np.mean(z_bin_heights_sorted[:5]) # take average of top 5 values
            z_heights_avg.append(z_height_max_avg)
            ax0.plot(tbt_dict['z_bin_centers'], tbt_dict['z_bin_heights'][:, i] / z_height_max_avg, label=plot_str[j])
            j += 1
        ax0.set_xlabel('$\zeta$ [m]')
        ax0.set_ylabel('Normalized counts')
        ax0.legend(loc='upper left', fontsize=14)
        plt.tight_layout()
        fig0.savefig('output_plots/SPS_Zeta_Beam_Profile_WS.png', dpi=250)
        
        #### Also generate plots comparing with profile measurements
        if also_compare_with_profile_data:
            # Load data, also after the RF spill
            zeta_SPS_inj, zeta_SPS_final, zeta_PS_BSM, data_SPS_inj, data_SPS_final, data_PS_BSM = self.load_longitudinal_profile_data()
            zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill = self.load_longitudinal_profile_after_SPS_injection_RF_spill()
    
            # Plot longitudinal phase space, initial and final state
            fig, ax = plt.subplots(2, 1, figsize = (8, 10), sharex=True)
            
            #### Simulated initial distribution
            ax[0].plot(tbt_dict['z_bin_centers'], tbt_dict['z_bin_heights'][:, index_to_plot[0]] / z_heights_avg[0], 
                       alpha=0.8, color='darkturquoise', label='Simulated inital')
            
            ### Measured injection profile, after or before initial RF spill
            if inj_profile_is_after_RF_spill:
                ax[0].plot(zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill, label='SPS wall current\nmonitor data\nafter RF capture')  
            else:
                ax[0].plot(zeta_SPS_inj, data_SPS_inj, label='SPS wall current\nmonitor data at inj')  
                ax[0].plot(zeta_PS_BSM, data_PS_BSM, label='PS BSM data \nat extraction')
                
            #### Simulated final distribution
            ax[1].plot(tbt_dict['z_bin_centers'], tbt_dict['z_bin_heights'][:, index_to_plot[1]] / z_heights_avg[1], 
                      alpha=0.8, color='lime', label='Simulated final')
            
            #### Measured final distribution
            ax[1].plot(zeta_SPS_final, data_SPS_final, color='darkgreen', label='SPS wall current\nmonitor data\n(at ~22 s)')
            
            ax[0].legend(loc='upper right', fontsize=13)
            ax[1].legend(loc='upper right', fontsize=13)
            
            # Adjust axis limits and plot turn
            ax[0].set_xlim(-0.85, 0.85)
            ax[1].set_xlim(-0.85, 0.85)
            
            ax[0].text(0.02, 0.91, plot_str[0], fontsize=15, transform=ax[0].transAxes)
            ax[1].text(0.02, 0.91, plot_str[1], fontsize=15, transform=ax[1].transAxes)
            #ax[1].text(0.02, 0.85, 'Time = {:.2f} s'.format(full_data_turns_seconds_index[ind_final]), fontsize=12, transform=ax[1].transAxes)
                
            ax[1].set_xlabel(r'$\zeta$ [m]')
            ax[1].set_ylabel('Counts')
            ax[0].set_ylabel('Normalized count')
            ax[1].set_ylabel('Normalized count')
            plt.tight_layout()
            
            if inj_profile_is_after_RF_spill:
                fig.savefig('output_plots/SPS_Pb_longitudinal_profiles_vs_data_after_RF_spill.png', dpi=250)
            else:
                fig.savefig('output_plots/SPS_Pb_longitudinal_profiles_vs_data.png', dpi=250)
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


    def load_emittance_data(self, path : str = emittance_data_path) -> pd.DataFrame:
        """
        Loads measured emittance data from SPS MDs, processed with CCC miner
        https://github.com/ewaagaard/ccc_miner, returns pd.DataFrame
        
        Default date - 2023-10-16 with (Qx, Qy) = (26.3, 26.19) in SPS
        """
        
        # Load dictionary with emittance data
        try:
            with open(path, 'r') as fp:
                full_data = json.load(fp)
        except FileNotFoundError:
            print('\nFILE NOT FOUND - check input path!\n')
            return
        
        # Convert timestamp strings to datetime, and find spread
        full_data['TimestampX_datetime'] = pd.to_datetime(full_data['UTC_timestamp_X'])
        full_data['TimestampY_datetime'] = pd.to_datetime(full_data['UTC_timestamp_Y'])
        
        full_data['N_emitX_error'] = np.std(full_data['N_emittances_X'], axis=1)
        full_data['N_emitY_error'] = np.std(full_data['N_emittances_Y'], axis=1)
        
        # Only keep the average emittances, not full emittance tables
        #del full_data['N_emittances_X'], full_data['N_emittances_Y']
        
        df = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in full_data.items() ]))
        
        # Remove emittance data after ramping starts, i.e. from around 48.5 s
        df = df[df['Ctime_Y'] < 48.5]
        
        return df


    def load_Nb_data(self, path : str = Nb_data_path, index=0) -> pd.DataFrame:
        """
        Loads measured FBCT bunch intensity data from SPS MDs, processed with CCC miner
        https://github.com/ewaagaard/ccc_miner, returns pd.DataFrame
        
        Default date - 2023-10-16 with (Qx, Qy) = (26.3, 26.19) in SPS
        """
        # Load dictionary with emittance data
        try:
            with open(path, 'r') as fp:
                Nb_dict = json.load(fp)
        except FileNotFoundError:
            print('\nFILE NOT FOUND - check input path!\n')
            return
        
        # Create new dictionary wiht numpy arrays - divide by charge
        new_dict = {'ctime' : Nb_dict['ctime'],
                    'Nb' : np.array(Nb_dict['Nb1'])[:, index] / 82}
        
        # Create dataframe with four bunches
        df_Nb = pd.DataFrame(new_dict)
        df_Nb = df_Nb[(df_Nb['ctime'] < 46.55) & (df_Nb['ctime'] > 0.0)]
        
        return df_Nb


    def load_longitudinal_profile_data(self, 
                                       path : str = longitudinal_data_path,
                                       gamma = 7.33):
        """
        Load Pb longitudinal profile measured at SPS injection with wall current monitor 
        and at PS extraction with Bunch Shape Monitor (BSM)
        
        Returns:
        --------
        zeta_SPS_inj, data_SPS, zeta_PS_BSM, data_BSM : np.ndarray
            arrays containing longitudinal position and amplitude data of SPS and PS
        gamma : float
            relativistic gamma at injection (7.33 typical value at SPS injection)
        """
        # Load the data
        with open(path, 'rb') as f:
            time_SPS_inj = np.load(f)
            time_SPS_final = np.load(f)
            time_PS_BSM = np.load(f)
            data_SPS_inj = np.load(f)
            data_SPS_final = np.load(f)
            data_PS_BSM = np.load(f)

        # Convert time data to position data - use PS extraction energy for Pb
        beta = np.sqrt(1 - 1/gamma**2)
        zeta_SPS_inj = time_SPS_inj * constants.c * beta # Convert time units to length
        zeta_SPS_final = time_SPS_final * constants.c * beta # Convert time units to length
        zeta_PS_BSM = time_PS_BSM * constants.c * beta # Convert time units to length

        # Adjust to ensure peak is at maximum
        zeta_SPS_inj -= zeta_SPS_inj[np.argmax(data_SPS_inj)]
        zeta_SPS_final -= zeta_SPS_inj[np.argmax(data_SPS_inj)]
        zeta_PS_BSM -= zeta_SPS_inj[np.argmax(data_SPS_inj)] # adjust both accordingly

        # BSM - only include data up to the artificial ringing, i.e. at around zeta = 0.36
        ind = np.where(zeta_PS_BSM < 0.36)
        zeta_PS_BSM = zeta_PS_BSM[ind]
        data_PS_BSM = data_PS_BSM[ind]

        return zeta_SPS_inj, zeta_SPS_final, zeta_PS_BSM, data_SPS_inj, data_SPS_final, data_PS_BSM


    def load_longitudinal_profile_after_SPS_injection_RF_spill(self, 
                                                               path : str = longitudinal_data_path_after_RF_spill, 
                                                               gamma = 7.33):
        """
        Load Pb longitudinal profile measured at SPS injection with wall current monitor 
        but AFTER initial spill out of RF bucket
        
        Returns:
        --------
        zeta_SPS_inj, data_SPS
            arrays containing longitudinal position and amplitude data of SPS 
        gamma : float
            relativistic gamma at injection (7.33 typical value at SPS injection)
        """
        # Load the data
        with open(path, 'rb') as f:
            time_SPS_inj_after_RF_spill = np.load(f)
            data_SPS_inj_after_RF_spill = np.load(f)

        # Convert time data to position data - use PS extraction energy for Pb
        beta = np.sqrt(1 - 1/gamma**2)
        zeta_SPS_inj_after_RF_spill = time_SPS_inj_after_RF_spill * constants.c * beta # Convert time units to length

        # Adjust to ensure peak is at maximum
        zeta_SPS_inj_after_RF_spill -= zeta_SPS_inj_after_RF_spill[np.argmax(data_SPS_inj_after_RF_spill)]

        return zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill



    def load_bunch_length_data(self, path : str = bunch_length_data_path):
        """
        Load fitted bunch lengths (assuming either Gaussian or binomial profiles) from
        Pb longitudinal profile measured at SPS injection plateau with wall current monitor
        from 2016 studies of Timas and Hannes
        
        Returns:
        --------
        sigma_RMS_Gaussian, sigma_RMS_Binomial, sigma_RMS_Gaussian_in_m, sigma_RMS_Binomial_in_m, ctime : np.ndarrays
            arrays containing fitted RMS beam size for Gaussian and Binomial, first in ns and then in m, and corresponding cycle time
        """
        
        # Save the bunch length data
        with open(bunch_length_data_path, 'rb') as f:
            sigma_RMS_Gaussian = np.load(f)
            sigma_RMS_Binomial = np.load(f)
            sigma_RMS_qGaussian = np.load(f)
            sigma_RMS_Gaussian_in_m = np.load(f)
            sigma_RMS_Binomial_in_m = np.load(f)
            sigma_RMS_qGaussian_in_m = np.load(f)
            q_vals = np.load(f)
            q_error = np.load(f)
            ctime = np.load(f)
            
        return sigma_RMS_Gaussian_in_m, sigma_RMS_Binomial_in_m, sigma_RMS_qGaussian_in_m, q_vals, q_error, ctime

        

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
                    ax.plot(zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill, label='SPS wall current\nmonitor data after RF capture')    
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