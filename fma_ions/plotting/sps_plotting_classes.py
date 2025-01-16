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
import xtrack as xt

from ..beam_parameters import BeamParameters_SPS, BeamParameters_SPS_Oxygen, BeamParameters_SPS_Proton, BeamParameters_SPS_Binomial_2016_before_RF_capture
from ..sequences import SPS_sequence_maker
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


    def optics_at_WS(self):
        """Reads betx, bety and dx from json file if exists """
        try:
            seq_folder = Path(__file__).resolve().parent.joinpath('../../data/sps_sequences')
            with open('{}/SPS_WS_optics.json'.format(seq_folder)) as f:
                optics_dict = json.load(f)
            
            # Select correct optics for the relevant wire scanner
            optics_ws_X = optics_dict['51637']
            optics_ws_Y = optics_dict['41677']
            
            return optics_ws_X['betx'], optics_ws_Y['bety'], optics_ws_X['dx'] 
        
        except FileNotFoundError:
            print('Optics file not found: need to run find_WS_optics module in data folder first!')


    def get_horizontal_aperture_size(self, el):
        """Convenience function to compute horizontal aperture size and beam sizes"""
        if hasattr(el, 'min_x'):
            return el.min_x, el.max_x
        if hasattr(el, 'max_x'):
            return -el.max_x, el.max_x
        return -el.a, el.a

    def get_vertical_aperture_size(self, el):
        """Convenience function to compute vertical aperture size and beam sizes"""
        if hasattr(el, 'min_y'):
            return el.min_y, el.max_y
        if hasattr(el, 'max_y'):
            return -el.max_y, el.max_y
        return -el.a, el.a
    
    def get_aperture(self, line, twiss):
        """Aperture from twiss table"""
        tt = line.get_table()
        survey = line.survey()
        apertypes = ['LimitEllipse', 'LimitRect', 'LimitRectEllipse', 'LimitRacetrack']
        aper_idx = np.where([tt['element_type', nn] in apertypes for nn in survey.name])[0]
        
        tw_ap = twiss.rows[aper_idx]
        sv_ap = survey.rows[aper_idx]
        apX_extent = np.array([self.get_horizontal_aperture_size(line[nn]) for nn in tw_ap.name])
        apX_offset = np.array([line[nn].shift_x for nn in tw_ap.name])
        apY_extent = np.array([self.get_vertical_aperture_size(line[nn]) for nn in tw_ap.name])
        apY_offset = np.array([line[nn].shift_y for nn in tw_ap.name])
        
        upperX = apX_offset + apX_extent[:, 0]
        lowerX = apX_offset + apX_extent[:, 1]
        upperY = apY_offset + apY_extent[:, 0]
        lowerY = apY_offset + apY_extent[:, 1]

        return sv_ap, tw_ap, upperX, lowerX, upperY, lowerY, aper_idx


    def plot_beam_envelope_and_aperture(self, n_sigmas=5, sigma_delta = 5e-4, add_beta_beat=False):
        """Method to plot beam envope for given sigma, and aperture"""
        
        # Load default line, with aperture
        sps_seq = SPS_sequence_maker()
        line, _ = sps_seq.load_xsuite_line_and_twiss() 
        
        if add_beta_beat:
            line = sps_seq.add_beta_beat_to_line(line)

        twiss = line.twiss()
        sv_ap, tw_ap, upperX, lowerX, upperY, lowerY, aper_idx = self.get_aperture(line, twiss)

        # Find beam parameters
        beamParams =  BeamParameters_SPS()
        sigx = np.sqrt(beamParams.exn / twiss.gamma0 * twiss.betx) + abs(twiss.dx) * sigma_delta
        sigy = np.sqrt(beamParams.eyn / twiss.gamma0 * twiss.bety)
        n_sigx_aper = lowerX / sigx[aper_idx]
        n_sigy_aper = lowerY / sigy[aper_idx]
        
        # X aperture #
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        ax.fill_between(tw_ap.s, upperX, lowerX, alpha=1., color='lightgrey')
        ax.plot(sv_ap.s, upperX, color="k")
        ax.plot(sv_ap.s, lowerX, color="k")
        ax.set_ylabel('x [m]')
        ax.set_xlabel('s [m]')
        ax.fill_between(twiss.s, twiss.x - n_sigmas * sigx, twiss.x + n_sigmas * sigx, alpha=0.5, color='blue')
        
        # Y aperture #
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        ax2.fill_between(tw_ap.s, upperY, lowerY, alpha=1., color='lightgrey')
        ax2.plot(sv_ap.s, upperY, color="k")
        ax2.plot(sv_ap.s, lowerY, color="k")
        ax2.set_ylabel('y [m]')
        ax2.set_xlabel('s [m]')
        ax2.fill_between(twiss.s, twiss.y - n_sigmas * sigy, twiss.y + n_sigmas * sigy, alpha=0.5, color='red')
        
        # Available sigmas to aperture
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        ax3.plot(sv_ap.s, n_sigx_aper, color="blue", label='X')
        ax3.plot(sv_ap.s, n_sigy_aper, color="red", label='Y')
        ax3.grid(alpha=0.5)
        ax3.set_ylabel('Available $\sigma_{x,y}$ to aperture')
        ax3.set_xlabel('s [m]')
        ax3.legend(fontsize=10)

        plt.show()

    def plot_tracking_data(self, 
                           tbt_dict=None, 
                           output_folder=None,
                           x_unit_in_turns=False,
                           distribution_type='qgaussian',
                           emittance_dict=None,
                           fbct_dict=None,
                           BL_dict=None,
                           inj_profile_is_after_RF_spill=True,
                           also_plot_sigma_delta=False,
                           plot_2016_bunch_length_measurements=False,
                           also_plot_2023_WCM_Nb_data=False,
                           also_plot_2023_DCBCT_Nb_data=False,
                           include_2023_emittance_measurements=False,
                           plot_emittances_separately=False,
                           also_plot_particle_std_BL=False,
                           return_fig=False):
        """
        Generates emittance plots from turn-by-turn (TBT) data class from simulations,
        compare with emittance measurements (default 2023-10-16) if desired.
        
        Parameters:
        tbt_dict : dict
            dictionary containing the TBT data. If None, loads json file.
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        x_units_in_turns : bool
            if True, x axis units will be turn, otherwise in seconds
        emittance_dict : dict
            Measured WS data. If not None, will plot measurement arrays with 'ctime', 'exn', 'dexn', 'eyn', 'deyn' with 'label'
        fbct_dict : dict
            Measued FBCT data. If not None, will plot arrays with 'ctime', 'Nb' with 'label'
        BL_dict : dict 
            Measued Wall Current Monitor bunch length data. If not None, will plot arrays with 'ctime', 'sigma_RMS' with 'label'
        distribution_type : str
            either 'qgaussian', 'gaussian' or 'binomial'
        inj_profile_is_after_RF_spill : bool
            whether SPS injection profile is after the initial spill out of the RF bucket
        also_plot_sigma_delta : bool
            whether also to plot sigma_delta
        plot_2016_bunch_length_measurements : bool
            whether to include bunch length measurements from SPS wall current monitor from 2016 studies by Hannes and Tomas
        also_plot_2023_WCM_Nb_data : bool
            whether to also plot Wall current monitor data
        also_plot_2023_DCBCT_Nb_data : bool
            whether to also plot DCBCT data
            started without considering initial RF spill, 0.95 means that the beam parameters were adjusted to after the spill
        include_2023_emittance_measurements : bool
            whether to include measured emittance or not
        also_plot_particle_std_BL : bool
            whether to also plot the standard deviation of particle zeta, i.e. discrete bunch length
        return_fig : bool
            whether to return figure and axis object - if False, will do plt.show()
        """
        os.makedirs('output_plots', exist_ok=True)
        
        if tbt_dict is None:
            tbt_dict = self.load_records_dict_from_json(output_folder=output_folder)

        # If bunch length measurements present, need to plot in seconds
        if plot_2016_bunch_length_measurements:
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
        if include_2023_emittance_measurements:
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
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (9.5, 3.6), constrained_layout=True)
        ax1.plot(time_units, tbt_dict['exn'] * 1e6, alpha=0.7, c='turquoise', lw=1.5, label='Simulated')
        ax2.plot(time_units, tbt_dict['eyn'] * 1e6, alpha=0.7, c='turquoise', lw=1.5, label='Simulated')
        
        # Include default 2023 measurements
        if include_2023_emittance_measurements:
            ax1.errorbar(time_units_x, 1e6 * np.array(full_data['N_avg_emitX']), yerr=1e6 * full_data['N_emitX_error'], 
                       color='blue', fmt="o", label="Measured")
            ax2.errorbar(time_units_y, 1e6 * np.array(full_data['N_avg_emitY']), yerr=1e6 * full_data['N_emitY_error'], 
                       color='darkorange', fmt="o", label="Measured")
            
        # Any custom emittance measurements
        if emittance_dict is not None:
            ax1.errorbar(emittance_dict['ctime'], emittance_dict['exn'], yerr = emittance_dict['dexn'], 
                       color='red', fmt="o", label = emittance_dict['label'])
            ax2.errorbar(emittance_dict['ctime'], emittance_dict['eyn'], yerr = emittance_dict['deyn'], 
                       color='red', fmt="o", label = emittance_dict['label'])
            
        # Plot bunch intensities, also with mme
        ax3.plot(time_units, tbt_dict['Nb'], alpha=0.7, lw=2.2, c='turquoise', label='Simulated')
        
        # Any custom FBCT measurements
        if fbct_dict is not None:
            ax3.plot(fbct_dict['ctime'], fbct_dict['Nb'], label=fbct_dict['label'], alpha=0.8, color='r')
        
        # Possibly include 2023 data
        if also_plot_2023_DCBCT_Nb_data:
            ax3.plot(time_units_DCBCT, Nb_BCT_normalized, label='DC-BCT', alpha=0.8, color='b')
        if also_plot_2023_WCM_Nb_data:
            beamParams = BeamParameters_SPS_Binomial_2016_before_RF_capture() # load nominal bunch intensity before RF capture
            ax3.plot(time_units_WCM, Nb_WCM * beamParams.Nb,  alpha=0.8,
                      label='Measured', color='r')

        # Find min and max emittance values - set window limits 
        all_emit = np.concatenate((tbt_dict['exn'], tbt_dict['eyn']))
        if include_2023_emittance_measurements:
            all_emit = np.concatenate((all_emit, np.array(full_data['N_avg_emitX']), np.array(full_data['N_avg_emitY'])))


        for a in [ax1, ax2, ax3]:
            a.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        ax1.set_ylabel(r'$\varepsilon_{x}^{n}$ [$\mu$m]')
        ax2.set_ylabel(r'$\varepsilon_{y}^{n}$ [$\mu$m]')
        ax3.set_ylabel(r'Ions per bunch $N_{b}$')

        ax1.legend(fontsize=12.1)
        ax1.set_ylim(0.0, 3.0)
        ax2.set_ylim(0.0, 3.0)
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
        elif distribution_type=='qgaussian':
            turn_array, time_array, sigmas_q_gaussian, q, q_error = self.fit_bunch_lengths_to_data(tbt_dict=tbt_dict, distribution=distribution_type)
        elif distribution_type=='binomial':
            turn_array, time_array, sigmas_binomial, m, m_error = self.fit_bunch_lengths_to_data(tbt_dict=tbt_dict, distribution=distribution_type)

        f3, ax22 = plt.subplots(1, 1, figsize = (8,6))
        # Uncomment if want to plot standard deviation of numerical particle object
        if also_plot_particle_std_BL:
            ax22.plot(time_units, tbt_dict['bunch_length'], color='darkcyan', alpha=0.7, lw=1.5, label='STD($\zeta$) of simulated particles')      
        
        if distribution_type=='gaussian':
            ax22.plot(turn_array if x_unit_in_turns else time_array, sigmas_gaussian, color='cyan', ls='--', alpha=0.95,
                      label='Simulated profiles')
            if plot_2016_bunch_length_measurements:
                ax22.plot(ctime, sigma_RMS_Gaussian_in_m, color='darkorange', label='Measured profiles')

        elif distribution_type=='binomial':
            ax22.plot(turn_array if x_unit_in_turns else time_array, sigmas_binomial, color='cyan', ls='--', alpha=0.95,
                      label='Simulated profiles')
            if plot_2016_bunch_length_measurements:
                ax22.plot(ctime, sigma_RMS_Binomial_in_m, color='darkorange', alpha=0.95, label='Measured profiles')
        elif distribution_type=='qgaussian':
            ax22.plot(turn_array if x_unit_in_turns else time_array, sigmas_q_gaussian, color='cyan', ls='--', alpha=0.95,
                      label='Simulated profiles')
            if plot_2016_bunch_length_measurements:
                ax22.plot(ctime, sigma_RMS_qGaussian_in_m, color='darkorange', alpha=0.95, label='Measured profiles')
                    
        # Any custom BL measurements
        if BL_dict is not None:
            ax22.plot(BL_dict['ctime'], BL_dict['sigma_RMS'], label=BL_dict['label'], alpha=0.85, color='r')

        ax22.set_ylabel(r'$\sigma_{{z, RMS}}$ [m] of {}'.format(distribution_type))
        ax22.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        ax22.legend()
        
        if distribution_type == 'qgaussian':
            # Insert extra box with fitted q-value of profiles - plot every 10th value
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
            
            # Add measured data
            if plot_2016_bunch_length_measurements:
                ax23.errorbar(ctime[start_ind::15], q_measured[start_ind::15], yerr=dq_measured[start_ind::15], markerfacecolor='darkorange', color='darkorange', alpha=0.65, ls='None', marker='o', ms=5.1, label='Measured')
            if BL_dict is not None:
                ax23.errorbar(BL_dict['ctime'][::15], BL_dict['q'][::15], yerr=BL_dict['q_error'][::15], marker='o', label=BL_dict['label'], alpha=0.75, color='r')
            
            ax23.set_ylabel('Fitted $q$-value', fontsize=13.5) #, color='green')
            #ax23.legend(fontsize=11, loc='upper left')
            
            ax23.tick_params(axis="both", labelsize=12)
            #ax23.tick_params(colors='green', axis='y')
            ax23.set_ylim(min(q)-0.2, max(q)+0.2)
            ax23.set_xlabel('Time [s]', fontsize=13.5)

        
        f3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        f3.savefig('output_plots/sigma_rms_and_qvalues.png', dpi=250)

        if return_fig:
            return f, (ax1, ax2, ax3) 
        else:
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

        output_folder_str = output_folder + '/' if output_folder is not None else ''

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
            with open('{}saved_bunch_length_fits.pickle'.format(output_folder_str), 'rb') as handle:
                BL_dict = pickle.load(handle)
                
            if distribution=='qgaussian':
                sigmas_q_gaussian= BL_dict['sigmas_q_gaussian']
                q_vals, q_errors = BL_dict['q_vals'], BL_dict['q_errors']
            elif distribution=='binomial':
                 sigmas_binomial = BL_dict['sigmas_binomial']
                 m, m_error = BL_dict['m'], BL_dict['m_error']
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
                            
                if distribution=='qgaussian':
                        # Fit both q-Gaussian and binomial
                        popt_Q, pcov_Q = fits.fit_Q_Gaussian(xdata, ydata)
                        q_vals[i] = popt_Q[1]
                        q_errors[i] = np.sqrt(np.diag(pcov_Q))[1] # error from covarance_matrix
                        sigmas_q_gaussian[i] = fits.get_sigma_RMS_from_qGaussian_fit(popt_Q)
                        print('Profile {}: q-Gaussian fit q={:.3f} +/- {:.2f}, sigma_RMS = {:.3f} m'.format(i, q_vals[i], q_errors[i], sigmas_q_gaussian[i]))
                elif distribution=='binomial':
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
                BL_dict = {'sigmas_q_gaussian': sigmas_q_gaussian, 'q_vals': q_vals, 'q_errors': q_errors}
            elif distribution=='binomial':
                BL_dict = {'sigmas_binomial': sigmas_binomial, 'm': m, 'm_error': m_error}
            else:
                BL_dict = {'sigmas': sigmas}
                    
            # Dump saved fits in dictionary, then pickle file
            with open('{}saved_bunch_length_fits.pickle'.format(output_folder_str), 'wb') as handle:
                pickle.dump(BL_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Dictionary with saved fits dumped')               

        #### First final profile vs fit
        if show_final_profile:
            fig0, ax0 = plt.subplots(1, 1, figsize = (8, 6))
            ax0.plot(xdata, ydata, label='Fit')
            if distribution=='qgaussian':
                ax0.plot(xdata, fits.Q_Gaussian(xdata, *popt_Q), color='green', ls='--', lw=2.8, label='q-Gaussian fit')
            elif distribution=='binomial':
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
        elif distribution=='qgaussian':
            return turn_array, time_array, sigmas_q_gaussian, q_vals, q_errors
        elif distribution=='binomial':
            return turn_array, time_array, sigmas_binomial, m, m_error


    def fit_and_plot_transverse_profiles(self, 
                                         output_str_array : list,
                                         scan_array_for_x_axis : np.ndarray,
                                         label_for_x_axis : str,
                                         extra_text_string,
                                         transmission_range=[0.0, 105],
                                         emittance_range = [0.0, 4.1],
                                         master_job_name=None,
                                         load_measured_profiles=False,
                                         x_axis_quantity='Qx',
                                         apply_uniform_xscale=False) -> None:

        """
        Open tbt data from e.g tune scan, plot transverse profiles and fit a q-Gaussian.
        Also generate loss plots across the

        Parameters:
        -----------
        output_str_array : [outfolder, outfolder, ...]
            List containing string for outfolder tbt data
        scan_array_for_x_axis : np.ndarray
            numpy array with quantity scanned over (e.g. Qx, Qy)
        label_for_x_axis : str
        extra_text_string : str
            plot extra text in bottom left corner
        transmission_range : list
            in which range to plot transmission
        emittance_range : list
            in which range to plot emittances
        load_measured_profiles : bool
            whether to plot measured profiles as well
        x_axis_quantity: str
            which quantity to use for x-axis, e.g. Qx or Qy
        apply_uniform_xscale : bool
            whether to force x axis scaling to be uniform, i.e. equal spacing for any value. Useful if want to plot uniformly values such as 1, 10 and 1000
        """
        # Generate directories, if not existing already
        os.makedirs('output_transverse', exist_ok=True)
        os.makedirs('output_transverse/X_profiles', exist_ok=True)
        os.makedirs('output_transverse/Y_profiles', exist_ok=True)
        os.makedirs('output_transverse/losses', exist_ok=True)

        # Initiate fit function
        fits = Fit_Functions()

        # Create empty arrays
        n_profiles = len(output_str_array)
        sigmas_q_gaussian_X = np.zeros(n_profiles)
        q0_vals_X = np.zeros(n_profiles)
        q0_errors_X = np.zeros(n_profiles)
        q_vals_X = np.zeros(n_profiles)
        q_errors_X = np.zeros(n_profiles)
        
        sigmas_q_gaussian_Y = np.zeros(n_profiles)
        q0_vals_Y = np.zeros(n_profiles)
        q0_errors_Y = np.zeros(n_profiles)         
        q_vals_Y = np.zeros(n_profiles)
        q_errors_Y = np.zeros(n_profiles) 

        # Load TBT data and append bunch intensities and emittances
        exn = np.zeros([2, len(output_str_array)]) # rows are initial and final, columns for each run
        eyn = np.zeros([2, len(output_str_array)])
        Nb = np.zeros([2, len(output_str_array)])
        transmission = np.zeros(len(output_str_array))
        all_loss_strings = ''

        # Load dictionary and append values
        for i, output_folder in enumerate(output_str_array):
            self.output_folder = output_folder
            scan_string = '{} = {:.2f}'.format(label_for_x_axis, scan_array_for_x_axis[i])
            try:
                tbt_dict = self.load_records_dict_from_json(output_folder=output_folder)

                particles_f = tbt_dict['particles_f']
                twiss = tbt_dict['twiss']
                df_twiss = pd.DataFrame(twiss)
                
                fig_phase_space, fig01_phase_space, fig1_phase_space, fig2_lost_at_turn, fig3_lost_at_s, loss_string = self.plot_normalized_phase_space_from_tbt(particles_f,
                                                                                      extra_text_string=scan_string, df_twiss=df_twiss)
                fig_phase_space.savefig('output_transverse/losses/Norm_phase_space_{}.png'.format(output_folder), dpi=250)
                fig01_phase_space.savefig('output_transverse/losses/x_y_phase_space_{}.png'.format(output_folder), dpi=250)
                fig1_phase_space.savefig('output_transverse/losses/X_Y_norm_phase_space_{}.png'.format(output_folder), dpi=250)
                fig2_lost_at_turn.savefig('output_transverse/losses/Lost_at_turn_{}.png'.format(output_folder), dpi=250)
                fig3_lost_at_s.savefig('output_transverse/losses/Lost_at_s_{}.png'.format(output_folder), dpi=250)
                del fig_phase_space, fig2_lost_at_turn, fig3_lost_at_s
                all_loss_strings += '\n{}\n{}'.format(scan_string, loss_string)

                # Initial emittances and bunch intensities
                exn[0, i] =  tbt_dict['exn'][0]
                eyn[0, i] =  tbt_dict['eyn'][0]
                Nb[0, i]  =  tbt_dict['Nb'][0] # initial
                                
                # Plot simulated particle profile
                fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)
                fig2, ax2 = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)

                # Plot measured profiles if desired
                if load_measured_profiles:
                    try: 
                        with open('measured_output_bws/X_average_bws_profiles_{}_{:.2f}.npy'.format(x_axis_quantity, scan_array_for_x_axis[i]), 'rb') as f:
                            x_pos = np.load(f)
                            x_prof_avg = np.load(f)

                        # Convert to m, normalize height
                        x_pos *= 1e-3
                        x_measured_bin_heights_sorted = np.array(sorted(x_prof_avg, reverse=True))
                        x_measured_height_max_avg = np.mean(x_measured_bin_heights_sorted[:5]) # take average of top 3 values
                        x_prof_avg_norm = x_prof_avg / x_measured_height_max_avg

                        # Fit Gaussian, center the profile and re-adjust heights
                        popt_X_meas, _ = fits.fit_Gaussian(x_pos, x_prof_avg_norm, p0=(1.0, 0.0, 0.02))
                        x_pos -= popt_X_meas[1]
                        x_prof_avg_norm /= popt_X_meas[0]
                        ax.plot(x_pos, x_prof_avg_norm, ls='-', color='blue', label='Measured BWS')

                    except FileNotFoundError:
                        print('Could not open measured X BWS profile')

                    try: 
                        with open('measured_output_bws/Y_average_bws_profiles_{}_{:.2f}.npy'.format(x_axis_quantity, scan_array_for_x_axis[i]), 'rb') as f:
                            y_pos = np.load(f)
                            y_prof_avg = np.load(f)

                        # Convert to m, normalize height
                        y_pos *= 1e-3
                        y_measured_bin_heights_sorted = np.array(sorted(y_prof_avg, reverse=True))
                        y_measured_height_max_avg = np.mean(y_measured_bin_heights_sorted[:5]) # take average of top 3 values
                        y_prof_avg_norm = y_prof_avg / y_measured_height_max_avg
                        # Fit Gaussian, center the profile and re-adjust heights
                        popt_Y_meas, _ = fits.fit_Gaussian(y_pos, y_prof_avg_norm, p0=(1.0, 0.0, 0.02))
                        y_pos -= popt_Y_meas[1]
                        y_prof_avg_norm /= popt_Y_meas[0]
                        ax2.plot(y_pos, y_prof_avg_norm, ls='-', color='blue', label='Measured BWS')
                    except FileNotFoundError:
                        print('Could not open measured Y BWS profile')

                # Select index to plot, e.g last set of 100 turns
                index_to_plot = [0, -1] #[-1] #
                plot_str = ['Simulated first 100 turns', 'Simulated last 100 turns'] #['Simulated, last 100 turns']
                colors = ['blue', 'orange']

                for j, ind in enumerate(index_to_plot):
                    # Normalize bin heights
                    x_bin_heights_sorted = np.array(sorted(tbt_dict['monitorH_x_intensity'][ind], reverse=True))
                    x_height_max_avg = np.mean(x_bin_heights_sorted[:3]) # take average of top 3 values
                    X_pos_data = tbt_dict['monitorH_x_grid']
                    if ind == 0:
                        X0_profile_data = tbt_dict['monitorH_x_intensity'][ind] / x_height_max_avg
                    else:
                        X_profile_data = tbt_dict['monitorH_x_intensity'][ind] / x_height_max_avg
                        ax.plot(X_pos_data, X_profile_data, label=plot_str[j], color=colors[j])
                
                ax.set_xlabel('x [m]')
                ax.set_ylabel('Normalized counts')
                ax.set_ylim(0, 1.1)
                ax.text(0.02, 0.05, '{} = {:.2f}'.format(label_for_x_axis, scan_array_for_x_axis[i]), transform=ax.transAxes, fontsize=10.8)

                # Plot profile of particles
                for j, ind in enumerate(index_to_plot):
                    # Normalize bin heights
                    y_bin_heights_sorted = np.array(sorted(tbt_dict['monitorV_y_intensity'][ind], reverse=True))
                    y_height_max_avg = np.mean(y_bin_heights_sorted[:3]) # take average of top 3 values
                    Y_pos_data = tbt_dict['monitorV_y_grid']
                    if ind == 0:
                        Y0_profile_data = tbt_dict['monitorV_y_intensity'][ind] / y_height_max_avg ### if changes fast, take particle histogram instead
                        #particles_i = tbt_dict['particles_i']
                    else:
                        Y_profile_data = tbt_dict['monitorV_y_intensity'][ind] / y_height_max_avg
                        ax2.plot(Y_pos_data, Y_profile_data, label=plot_str[j], color=colors[j])
                
                ax2.set_ylabel('Normalized counts')
                ax2.set_xlabel('y [m]')
                ax2.set_ylim(0, 1.1)
                ax2.text(0.02, 0.05, '{} = {:.2f}'.format(label_for_x_axis, scan_array_for_x_axis[i]), transform=ax2.transAxes, fontsize=10.8)

                # Fit Gaussian for the emittance
                popt_X, pcov_X = fits.fit_Gaussian(X_pos_data, X_profile_data, p0=(1.0, 0.0, 0.02))
                popt_Y, pcov_Y = fits.fit_Gaussian(Y_pos_data, Y_profile_data, p0=(1.0, 0.0, 0.02))
                sigma_raw_X = np.abs(popt_X[2])
                sigma_raw_Y = np.abs(popt_Y[2])

                ### Convert beam sizes to emittances ###
                part = tbt_dict['particles_i']
                gamma = part['gamma0'][0]
                beta_rel = part['beta0'][0]

                # Fit q-Gaussian to final X and Y profiles, to latest curves - initial guess from Gaussian
                q0 = 1.02
                p0_qX = [popt_X[1], q0, 1/popt_X[2]**2/(5-3*q0), 2*popt_X[0]]
                p0_qY = [popt_Y[1], q0, 1/popt_Y[2]**2/(5-3*q0), 2*popt_Y[0]]

                popt_Q_X0, pcov_Q_X0 = fits.fit_Q_Gaussian(X_pos_data, X0_profile_data, p0=p0_qX)
                q0_vals_X[i] = popt_Q_X0[1]
                q0_errors_X[i] = np.sqrt(np.diag(pcov_Q_X0))[1] # error from covarance_matrix

                popt_Q_X, pcov_Q_X = fits.fit_Q_Gaussian(X_pos_data, X_profile_data, p0=p0_qX)
                q_vals_X[i] = popt_Q_X[1]
                q_errors_X[i] = np.sqrt(np.diag(pcov_Q_X))[1] # error from covarance_matrix
                sigmas_q_gaussian_X[i] = fits.get_sigma_RMS_from_qGaussian_fit(popt_Q_X)

                popt_Q_Y, pcov_Q_Y = fits.fit_Q_Gaussian(Y_pos_data, Y_profile_data, p0=p0_qY)
                q_vals_Y[i] = popt_Q_Y[1]
                q_errors_Y[i] = np.sqrt(np.diag(pcov_Q_Y))[1] # error from covarance_matrix
                
                popt_Q_Y0, pcov_Q_Y0 = fits.fit_Q_Gaussian(Y_pos_data, Y0_profile_data, p0=p0_qY)
                q0_vals_Y[i] = popt_Q_Y0[1]
                q0_errors_Y[i] = np.sqrt(np.diag(pcov_Q_Y0))[1] # error from covarance_matrix
                
                sigmas_q_gaussian_Y[i] = fits.get_sigma_RMS_from_qGaussian_fit(popt_Q_Y)
                
                # Extract optics at Wire Scanner, correct for dispersion
                betx, bety, dx = self.optics_at_WS()
                dpp = 1e-3
                sigmaX_raw_for_betatronic = sigma_raw_X # sigmas_q_gaussian_X[i]
                sigmaX_betatronic = np.sqrt((sigmaX_raw_for_betatronic)**2 - (dpp * dx)**2)
                exf = sigmaX_betatronic**2 / betx

                sigmaY_raw_for_betatronic = sigma_raw_Y # sigmas_q_gaussian_Y[i]
                sigmaY_betatronic = np.abs(sigmaY_raw_for_betatronic) # no vertical dispersion
                eyf = sigmaY_betatronic**2 / bety
                exn[1, i] =  exf * beta_rel * gamma 
                eyn[1, i] =  eyf * beta_rel * gamma 
                Nb[1, i]  =  tbt_dict['Nb'][-1] # final
                transmission[i] = Nb[1, i]/Nb[0, i]

                print('X final profile: sigma = {:.3f} mm\nX q-Gaussian fit q={:.3f} +/- {:.2f}, sigma_RMS = {:.3f} mm'.format(sigma_raw_X*1e3, q_vals_X[i], q_errors_X[i], 1e3*sigmas_q_gaussian_X[i]))
                print('Y final profile: sigma = {:.3f} mm\nY q-Gaussian fit q={:.3f} +/- {:.2f}, sigma_RMS = {:.3f} mm'.format(sigma_raw_Y*1e3, q_vals_Y[i], q_errors_Y[i], 1e3*sigmas_q_gaussian_Y[i]))

                print('exn = {:.3e}, eyn = {:.3e}'.format(exn[1, i], eyn[1, i]))
                print('Transmission = {:3f}\n'.format(transmission[i]))


                # Add q-Gaussian fits to plots and save 
                #ax.text(0.02, 0.65, 'Final simulated:\n$\epsilon_{{x}}^n$ = {:.3f} $\mu$m rad'.format(1e6 * exn[1, i], 1e6), fontsize=12.5, transform=ax.transAxes)
                #ax2.text(0.02, 0.65, 'Final simulated\n$\epsilon_{{y}}^n$ = {:.3f} $\mu$m rad'.format(1e6 * eyn[1, i]), fontsize=12.5, transform=ax2.transAxes)

                ax.plot(X_pos_data, fits.Q_Gaussian(X_pos_data, *popt_Q_X), ls='--', color='lime', label='q-Gaussian fit final\nsimulated profiles')
                ax2.plot(Y_pos_data, fits.Q_Gaussian(Y_pos_data, *popt_Q_Y), ls='--', color='lime', label='q-Gaussian fit final\nsimulated profiles')
                ax.legend(loc='upper left', fontsize=11.5)
                ax2.legend(loc='upper left', fontsize=11.5)
                fig.savefig('output_transverse/X_profiles/{}_SPS_X_Beam_Profile_WS.png'.format(output_folder), dpi=250)
                fig2.savefig('output_transverse/Y_profiles/{}_SPS_Y_Beam_Profile_WS.png'.format(output_folder), dpi=250)
                plt.close()


            except FileNotFoundError:
                print('Could not find values in {}!\n'.format(output_folder))
                exn[0, i] = np.nan
                exn[1, i] = np.nan
                eyn[0, i] = np.nan
                eyn[1, i] = np.nan
                Nb[0, i] = np.nan
                Nb[1, i] = np.nan
                q_vals_X[i] = np.nan
                q_vals_Y[i] = np.nan
                q0_vals_X[i] = np.nan
                q0_vals_Y[i] = np.nan
                transmission[i] = np.nan
        
        # Save loss string to txt tile
        with open('output_transverse/large_losses.txt', 'w') as file:
            file.write(all_loss_strings)
        
        ### Plot transmissions and final q-Gaussian emittances - use custom x axis spacing
        fig3, ax3 = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, constrained_layout=True)
        if apply_uniform_xscale:
            xx = np.arange(len(scan_array_for_x_axis))
            xlabels = [str(x) for x in scan_array_for_x_axis]
        else:
            xx = scan_array_for_x_axis
        ax3[0].plot(xx, exn[1, :] * 1e6, c='b', marker="o", label="X - final")
        ax3[0].plot(xx, eyn[1, :] * 1e6, c='darkorange', marker="o", label="Y - final")
        ax3[0].plot(xx, exn[0, :] * 1e6, c='b', ls='--', lw=1.0, alpha=0.75, marker=".", label="X - initial")
        ax3[0].plot(xx, eyn[0, :] * 1e6, c='darkorange', ls='--', lw=1.0, alpha=0.75, marker=".", label="Y - initial")
        ax3[1].set_xticks(xx)
        if apply_uniform_xscale:
            ax3[1].set_xticklabels(xlabels)

        ax3[0].set_ylabel("$\epsilon_{x, y}^n$ [$\mu$m]")       
        ax3[1].plot(xx, 100*transmission, c='red', marker='o', label='Transmission')
        ax3[1].set_ylabel("Transmission [%]")
        ax3[0].legend(fontsize=13)
        ax3[0].grid(alpha=0.55)
        ax3[1].grid(alpha=0.55)
        ax3[1].tick_params(axis='x', which='major', rotation=35, labelsize=12.4)
        if extra_text_string is not None:
            ax3[1].text(0.024, 0.05, extra_text_string, transform=ax3[1].transAxes, fontsize=12.8)
        ax3[1].set_xlabel(label_for_x_axis)
        if emittance_range is not None:
            ax3[0].set_ylim(emittance_range[0], emittance_range[1])
        if transmission_range is not None:
            ax3[1].set_ylim(transmission_range[0], transmission_range[1])
        if master_job_name is None:
            master_job_name = 'scan_result_final_emittances_and_bunch_intensity'
        fig3.savefig('output_transverse/emittance_evolution_qGaussian_fits_{}.png'.format(master_job_name), dpi=250)
        plt.close()

        # Also plot q-values
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        if apply_uniform_xscale:
            ax1.errorbar(xx, y=q0_vals_X, yerr=q0_errors_X, c='b', fmt=".--", lw=1.0, alpha=0.75, label="Simulated $q_{X}$, first 100 turns")
            ax1.errorbar(xx, y=q0_vals_Y, yerr=q0_errors_Y, c='darkorange', fmt=".--", lw=1.0, alpha=0.75, label="Simulated $q_{Y}$ first 100 turns")
        ax1.errorbar(xx, y=q_vals_X, yerr=q_errors_X, c='b', fmt="o-", label="Simulated $q_{X}$, last 100 turns")
        ax1.errorbar(xx, y=q_vals_Y, yerr=q_errors_Y, c='darkorange',  fmt="o-", label="Simulated $q_{Y}$, last 100 turns")
        ax1.set_xticks(xx)
        if apply_uniform_xscale:
            ax1.set_xticklabels(xlabels)
        
        #ax[0].axhline(y=1.5, c='darkgreen', label=None)   # at extr, not end of flat bottom
        ax1.set_ylim(0, 2.0)
        ax1.set_ylabel("Fitted $q_{x,y}$")
        ax1.grid(alpha=0.5)
        ax1.tick_params(axis='x', which='major', rotation=35, labelsize=12.4)
        ax1.set_xlabel(label_for_x_axis)
        ax1.legend(loc="upper left", fontsize=11.5)
        fig1.savefig('output_transverse/q_value_evolution_qGaussian_fits_{}.png'.format(master_job_name), dpi=250)
        plt.close()

        # Save values
        with open('output_transverse/simulated_emittances_transmissions_and_qvalues.npy', 'wb') as f:
            np.save(f, scan_array_for_x_axis)
            np.save(f, exn[1, :])
            np.save(f, eyn[1, :])
            np.save(f, exn[0, :])
            np.save(f, eyn[0, :])
            np.save(f, transmission)
            np.save(f, q_vals_X)
            np.save(f, q_vals_Y)
            np.save(f, q_errors_X)
            np.save(f, q_errors_Y)
            

    def fit_transverse_profile_evolution(self, profile_step=100, output_folder_name='transverse_profile_evolution',
                                         tbt_dict=None, output_folder=None):
        "Load turn-by-turn data and plot emittance evolution"
        
        # Generate directories, if not existing already
        os.makedirs(output_folder_name, exist_ok=True)

        # Initiate fit function
        fits = Fit_Functions()
        # Extract optics at Wire Scanner, correct for dispersion
        betx, bety, dx = self.optics_at_WS()
        dpp = 1e-3

        # Load turn-by-turn data
        if tbt_dict is None:
            tbt_dict = self.load_records_dict_from_json(output_folder=output_folder)
        
        index_to_plot = np.arange(0, len(tbt_dict['monitorH_x_intensity'])+1, profile_step)
        index_to_plot[-1] -= 1 # correct counting index 
        turns_to_plot = 100*index_to_plot # 100 tursn for each index
        # Create empty arrays
        n_profiles = len(index_to_plot)
        exn = np.zeros(n_profiles)
        q_vals_X = np.zeros(n_profiles)
        q_errors_X = np.zeros(n_profiles)
        eyn = np.zeros(n_profiles)
        q_vals_Y = np.zeros(n_profiles)
        q_errors_Y = np.zeros(n_profiles) 
          
        for i, ind in enumerate(index_to_plot):
            print('Fitting WS profile turn {}'.format(turns_to_plot[i]))

            # Plot simulated particle profile
            fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)
            fig2, ax2 = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)

            # Normalize bin heights
            x_bin_heights_sorted = np.array(sorted(tbt_dict['monitorH_x_intensity'][ind], reverse=True))
            x_height_max_avg = np.mean(x_bin_heights_sorted[:3]) # take average of top 3 values
            X_pos_data = tbt_dict['monitorH_x_grid']
            X_profile_data = tbt_dict['monitorH_x_intensity'][ind] / x_height_max_avg
            ax.plot(X_pos_data, X_profile_data, label='Turn {}'.format(turns_to_plot[i]), color='orange')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('Normalized counts')
            ax.set_ylim(0, 1.1)

            # Normalize bin heights
            y_bin_heights_sorted = np.array(sorted(tbt_dict['monitorV_y_intensity'][ind], reverse=True))
            y_height_max_avg = np.mean(y_bin_heights_sorted[:3]) # take average of top 3 values
            Y_pos_data = tbt_dict['monitorV_y_grid']
            Y_profile_data = tbt_dict['monitorV_y_intensity'][ind] / y_height_max_avg
            ax2.plot(Y_pos_data, Y_profile_data, label='Turn {}'.format(turns_to_plot[i]), color='orange')
            ax2.set_ylabel('Normalized counts')
            ax2.set_xlabel('y [m]')
            ax2.set_ylim(0, 1.1)

            # Fit Gaussian for the emittance
            popt_X, pcov_X = fits.fit_Gaussian(X_pos_data, X_profile_data, p0=(1.0, 0.0, 0.02))
            popt_Y, pcov_Y = fits.fit_Gaussian(Y_pos_data, Y_profile_data, p0=(1.0, 0.0, 0.02))
            sigma_raw_X = np.abs(popt_X[2])
            sigma_raw_Y = np.abs(popt_Y[2])

            ### Convert beam sizes to emittances ###
            part = tbt_dict['particles_i']
            gamma = part['gamma0'][0]
            beta_rel = part['beta0'][0]

            # Fit q-Gaussian to final X and Y profiles, to latest curves - initial guess from Gaussian
            q0 = 1.02
            p0_qX = [popt_X[1], q0, 1/popt_X[2]**2/(5-3*q0), 2*popt_X[0]]
            p0_qY = [popt_Y[1], q0, 1/popt_Y[2]**2/(5-3*q0), 2*popt_Y[0]]

            popt_Q_X, pcov_Q_X = fits.fit_Q_Gaussian(X_pos_data, X_profile_data, p0=p0_qX)
            q_vals_X[i] = popt_Q_X[1]
            q_errors_X[i] = np.sqrt(np.diag(pcov_Q_X))[1] # error from covarance_matrix
            #sigmas_q_gaussian_X[i] = fits.get_sigma_RMS_from_qGaussian_fit(popt_Q_X)

            popt_Q_Y, pcov_Q_Y = fits.fit_Q_Gaussian(Y_pos_data, Y_profile_data, p0=p0_qY)
            q_vals_Y[i] = popt_Q_Y[1]
            q_errors_Y[i] = np.sqrt(np.diag(pcov_Q_Y))[1] # error from covarance_matrix
            #sigmas_q_gaussian_Y[i] = fits.get_sigma_RMS_from_qGaussian_fit(popt_Q_Y)

            sigmaX_raw_for_betatronic = sigma_raw_X # sigmas_q_gaussian_X[i]
            sigmaX_betatronic = np.sqrt((sigmaX_raw_for_betatronic)**2 - (dpp * dx)**2)
            exf = sigmaX_betatronic**2 / betx

            sigmaY_raw_for_betatronic = sigma_raw_Y # sigmas_q_gaussian_Y[i]
            sigmaY_betatronic = np.abs(sigmaY_raw_for_betatronic) # no vertical dispersion
            eyf = sigmaY_betatronic**2 / bety
            exn[i] =  exf * beta_rel * gamma 
            eyn[i] =  eyf * beta_rel * gamma 

            ax.plot(X_pos_data, fits.Q_Gaussian(X_pos_data, *popt_Q_X), ls='--', color='lime', label='q-Gaussian fit')
            ax2.plot(Y_pos_data, fits.Q_Gaussian(Y_pos_data, *popt_Q_Y), ls='--', color='lime', label='q-Gaussian fit')
            ax.legend(loc='upper left', fontsize=11.5)
            ax2.legend(loc='upper left', fontsize=11.5)
            fig.savefig('{}/X_profile_turn_{}.png'.format(output_folder_name, turns_to_plot[i]), dpi=250)
            fig2.savefig('{}/Y_profile_turn_{}.png'.format(output_folder_name, turns_to_plot[i]), dpi=250)

            del fig, fig2
            plt.close()

        # Plot fitted emittances and bunch intensity 
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (9.5, 3.6), constrained_layout=True)
        ax1.plot(turns_to_plot, exn * 1e6, alpha=0.7, c='turquoise', marker='o', ls='None', label='Simulated WS profiles')
        ax2.plot(turns_to_plot, eyn * 1e6, alpha=0.7, c='turquoise',  marker='o', ls='None', label='Simulated WS profiles')
        ax3.plot(tbt_dict['Turns'], tbt_dict['Nb'], alpha=0.7, lw=2.2, c='turquoise', label='Simulated')
        for a in [ax1, ax2, ax3]:
            a.set_xlabel('Turns')
            a.grid(alpha=0.55)
        ax1.set_ylabel(r'Fitted $\varepsilon_{x}^{n}$ [$\mu$m]')
        ax2.set_ylabel(r'Fitted $\varepsilon_{y}^{n}$ [$\mu$m]')
        ax3.set_ylabel(r'Ions per bunch $N_{b}$')
        ax1.legend(fontsize=12.1)
        ax1.set_ylim(0.0, 4.05)
        ax2.set_ylim(0.0, 4.05)
        ax3.set_ylim(0.0, max(tbt_dict['Nb'])*1.05)
        f.savefig('{}/0000_epsilon_Nb.png'.format(output_folder_name), dpi=250)

        # Also plot q-values
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        ax1.errorbar(turns_to_plot, y=q_vals_X, yerr=q_errors_X, fmt="o-", label="Simulated $q_{X}$")
        ax1.errorbar(turns_to_plot, y=q_vals_Y, yerr=q_errors_Y, fmt="o-", label="Simulated $q_{Y}$")
        #ax[0].axhline(y=1.5, c='darkgreen', label=None)   # at extr, not end of flat bottom
        ax1.set_ylim(0, 2.0)
        ax1.set_ylabel("Fitted $q_{x,y}$")
        ax1.grid(alpha=0.5)
        ax1.set_xlabel('Turns')
        ax1.legend(loc="upper left", fontsize=11.5)
        fig1.savefig('{}/0001_q_value_evolution_qGaussian_fits.png'.format(output_folder_name), dpi=250)
        plt.close()

        return turns_to_plot, exn, eyn


    def plot_multiple_sets_of_tracking_data(self, 
                                            output_str_array, 
                                            string_array, 
                                            compact_mode=False,
                                            include_emittance_measurements=False, 
                                            emittance_limits=None,
                                            plot_bunch_length=False,
                                            plot_bunch_length_measurements=False,
                                            x_unit_in_turns=False,
                                            bbox_to_anchor_position=(0.0, 1.3),
                                            labelsize = 15.8,
                                            ylim=None, 
                                            legend_font_size=11.4,
                                            extra_str='',
                                            also_plot_WCM_Nb_data=False,
                                            adjusting_factor_Nb_for_initial_drop=0.95,
                                            distribution_type='qgaussian',
                                            also_plot_particle_std_BL=False,
                                            bunch_length_in_log_scale=False,
                                            ylim_bunch_length=None, 
                                            return_fig=False):
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
        plot_bunch_length : bool
            whether to plot bunch length measurements or not
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
        also_plot_particle_std_BL : bool
            whether to plot standard deviation of particle zeta ("discrete" bunch length)
        ylim_bunch_length : list
            lower and upper bounds for emittance plots, if None (default), automatic limits are set
        return_fig : bool
            whether to return figure and axis object - if False, will do plt.show()
        """
        os.makedirs('main_plots', exist_ok=True)

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
            ax1.text(0.04, 0.91, '{}'.format(extra_str), fontsize=15, transform=ax1.transAxes)
            f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            f.savefig('main_plots/result_multiple_trackings{}.png'.format(extra_str), dpi=250)
            
            
            # Bunch length
            if plot_bunch_length:
                f3, ax22 = plt.subplots(1, 1, figsize = (8,6))
                for i, tbt_dict in enumerate(tbt_array):
                    if distribution_type=='gaussian':
                        turn_array, time_array, sigmas_gaussian = self.fit_bunch_lengths_to_data(tbt_dict=tbt_dict, distribution=distribution_type, 
                                                                                                 output_folder=output_str_array[i])
                    else:
                        turn_array, time_array, sigmas_q_gaussian, sigmas_binomial, q, q_error, m, m_error = self.fit_bunch_lengths_to_data(tbt_dict=tbt_dict,
                                                                                                            distribution=distribution_type,
                                                                                                            output_folder=output_str_array[i])
                    
                    # Uncomment if want to plot standard deviation of numerical particle object
                    if also_plot_particle_std_BL:
                        ax22.plot(time_units, tbt_dict['bunch_length'],  alpha=0.7, lw=1.5, label=string_array[i] + ' STD($\zeta$)')
    
                    if distribution_type=='gaussian':
                        ax22.plot(turn_array if x_unit_in_turns else time_array, sigmas_gaussian, ls='--', alpha=0.95,
                                  label='Simulated profiles')
                    elif distribution_type=='binomial':
                        ax22.plot(turn_array if x_unit_in_turns else time_array, sigmas_binomial, ls='--', alpha=0.95,
                                  label='Simulated profiles')
                    elif distribution_type=='qgaussian':
                        ax22.plot(turn_array if x_unit_in_turns else time_array, sigmas_q_gaussian, ls='--', alpha=0.95,
                                  label=string_array[i])
                        print('{}: relative bunch length increase = {:.3f} %'.format(string_array[i], 
                                                                                   100 * (sigmas_q_gaussian[-1] - sigmas_q_gaussian[0]) / sigmas_q_gaussian[0]))
                if plot_bunch_length_measurements:
                    # Load bunch length data
                    sigma_RMS_Gaussian_in_m, sigma_RMS_Binomial_in_m, sigma_RMS_qGaussian_in_m, q_vals, q_error, ctime = self.load_bunch_length_data()
                    ax22.plot(ctime, sigma_RMS_qGaussian_in_m, color='orangered', alpha=0.7, ls='--', label='Measured RMS q-Gaussian')
                if ylim_bunch_length is not None:
                    ax22.set_ylim(ylim_bunch_length[0], ylim_bunch_length[1])
                if bunch_length_in_log_scale:
                    ax22.set_yscale('log')
                ax22.set_ylabel(r'$\sigma_{z}$ [m]')
                ax22.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
                ax22.legend(fontsize=legend_font_size, loc='upper left')
                f3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                f3.savefig('main_plots/sigma_multiple_trackings{}.png'.format(extra_str), dpi=250)
            
            
        if return_fig:
            return f, (ax1, ax2, ax3) 
        else:
            plt.show()


    def plot_final_emittances_and_Nb_over_scan(self,
                                               scan_array_for_x_axis : np.ndarray,
                                               label_for_x_axis : str,
                                               output_str_array,
                                               extra_text_string=None,
                                               transmission_range=[0.0, 105],
                                               emittance_range = [0.0, 4.1],
                                               plot_starting_emittances=True,
                                               master_job_name=None) -> None:
        """
        Method to plot emittances and bunch intensities (final vs initial) for a 

        Parameters:
        -----------
        scan_array_for_x_axis : np.ndarray
            numpy array with quantity scanned over (e.g. Qx, Qy)
        label_for_x_axis : str
        output_str_array : [outfolder, outfolder, ...]
            List containing string for outfolder tbt data
        extra_text_string : str
            plot extra text in bottom left corner
        transmission_range : list
            in which range to plot transmission
        emittance_range : list
            in which range to plot emittances
        plot_starting_emittances : bool
            whether to plot flat line with starting emittances
        master_job_name : str
            which name the plot should have. Default is None, then default name is given
        """
        os.makedirs('output', exist_ok=True)
        # Load TBT data and append bunch intensities and emittances
        exn = np.zeros([2, len(output_str_array)]) # rows are initial and final, columns for each run
        eyn = np.zeros([2, len(output_str_array)])
        Nb = np.zeros([2, len(output_str_array)])
        transmission = np.zeros(len(output_str_array))

        # Load dictionary and append values
        for i, output_folder in enumerate(output_str_array):
            self.output_folder = output_folder
            try:
                tbt_dict = self.load_records_dict_from_json(output_folder=output_folder)

                # Append emittance and intensity values, initial and final
                exn[0, i] =  tbt_dict['exn'][0] # initial
                exn[1, i] =  tbt_dict['exn'][-1] # final
                eyn[0, i] =  tbt_dict['eyn'][0] # initial
                eyn[1, i] =  tbt_dict['eyn'][-1] # final
                Nb[0, i]  =  tbt_dict['Nb'][0] # initial
                Nb[1, i]  =  tbt_dict['Nb'][-1] # final
                transmission[i] = Nb[1, i]/Nb[0, i]
            except FileNotFoundError:
                print('Could not find values in {}!'.format(output_folder))
                exn[0, i] = np.nan
                exn[1, i] = np.nan
                eyn[0, i] = np.nan
                eyn[1, i] = np.nan
                Nb[0, i] = np.nan
                Nb[1, i] = np.nan
                transmission[i] = np.nan

        # Plot transmission and final emittances
        # Plot the transmission with emittance - ion tunes
        fig, ax = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, constrained_layout=True)
        ax[0].plot(scan_array_for_x_axis, exn[1, :] * 1e6, c='b', marker="o", label="X - final")
        ax[0].plot(scan_array_for_x_axis, eyn[1, :] * 1e6, c='darkorange', marker="o", label="Y - final")
        if plot_starting_emittances:
            ax[0].plot(scan_array_for_x_axis, exn[0, :] * 1e6, c='b', ls='--', lw=1.0, alpha=0.75, marker=".", label="X - initial")
            ax[0].plot(scan_array_for_x_axis, eyn[0, :] * 1e6, c='darkorange', ls='--', lw=1.0, alpha=0.75, marker=".", label="Y - initial")

        ax[0].set_ylabel("$\epsilon_{x, y}^n$ [$\mu$m]")       
        ax[1].plot(scan_array_for_x_axis, 100*transmission, c='red', marker='o', label='Transmission')
        ax[1].set_ylabel("Transmission [%]")
        ax[0].legend(fontsize=13)
        ax[0].grid(alpha=0.55)
        ax[1].grid(alpha=0.55)
        ax[1].set_xticks(scan_array_for_x_axis)
        ax[1].tick_params(axis='x', which='major', rotation=35, labelsize=12.4)
        if extra_text_string is not None:
            ax[1].text(0.024, 0.05, extra_text_string, transform=ax[1].transAxes, fontsize=12.8)
        ax[1].set_xlabel(label_for_x_axis)
        if emittance_range is not None:
            ax[0].set_ylim(emittance_range[0], emittance_range[1])
        if transmission_range is not None:
            ax[1].set_ylim(transmission_range[0], transmission_range[1])
        if master_job_name is None:
            master_job_name = 'scan_result_final_emittances_and_bunch_intensity'
        fig.savefig('output/{}.png'.format(master_job_name), dpi=250)
        plt.show()


    def plot_normalized_phase_space_from_tbt(self, 
                                             part_dict,
                                             x_min_norm_aperture=0.004147391486397011,
                                             x_min_norm_aperture_loc=5569.7227,
                                             y_min_norm_aperture=0.003013704789098143,
                                             y_min_norm_aperture_loc=6886.404799999996,
                                             x_min_aperture=0.03,
                                             y_min_aperture=0.01615,
                                             extra_text_string='',
                                             df_twiss=None):
        """
        Generate normalized phase space in X and Y to follow particle distribution
        
        Parameters:
        -----------
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        part_dict : dict
            dictionary with particle data
        include_density_map : bool
            whether to add color gradient of how tightly packed particles are
        plot_min_aperture : bool
            whether to include line with minimum X and Y aperture along machine
        x_min_norm_aperture : float
            default minimum aperture in X
        x_min_norm_aperture_loc : float
            location of minimum X aperture
        y_min_norm_aperture : float
            default minimum aperture in Y
        y_min_norm_aperture_loc : float
            location of minimum Y aperture
        extra_text_string : str
            plot extra text in plots
        df_twiss : pd.DataFrame
            twiss table in pandas format
        """
        # Output directory
        #os.makedirs('output', exist_ok=True)

        # Convert particle lists to numpy
        for key in part_dict:
            if type(part_dict[key]) == list:
                part_dict[key] = np.array(part_dict[key])
        
        # Final dead and alive indices
        alive_ind_final = part_dict['state'] > 0
        dead_ind_final = part_dict['state'] < 1
        print('Loaded particle object: {} killed out of {}'.format(len(part_dict['x'][dead_ind_final]),
                                                                       len(part_dict['x'])))

        # Convert to normalized phase space
        sps = SPS_sequence_maker()
        if df_twiss is None:
            df_twiss = sps.load_default_twiss_table(cycled_to_minimum_dx=True) # load twiss table with aperture

        # ALIVE particles - find normalized particle coordinates at start of line
        X_alive = part_dict['x'][alive_ind_final] / np.sqrt(df_twiss['betx'][0]) 
        PX_alive = df_twiss['alfx'][0] / np.sqrt(df_twiss['betx'][0]) * part_dict['x'][alive_ind_final] + np.sqrt(df_twiss['betx'][0]) * part_dict['px'][alive_ind_final]
        Y_alive = part_dict['y'][alive_ind_final] / np.sqrt(df_twiss['bety'][0]) 
        PY_alive = df_twiss['alfy'][0] / np.sqrt(df_twiss['bety'][0]) * part_dict['y'][alive_ind_final] + np.sqrt(df_twiss['bety'][0]) * part_dict['py'][alive_ind_final]
        
        # DEAD particles - find normalized particle coordinates where last seen
        dead_part_s = part_dict['s'][dead_ind_final]
        
        # Find where in Twiss table dead particles were seen last
        ind_s_dead = []
        for s in dead_part_s:
            ind_s_dead.append(np.abs(df_twiss['s'] - s).argmin())

        X_dead = part_dict['x'][dead_ind_final] / np.sqrt(df_twiss['betx'][ind_s_dead]) 
        PX_dead = df_twiss['alfx'][ind_s_dead] / np.sqrt(df_twiss['betx'][ind_s_dead]) * part_dict['x'][dead_ind_final] + np.sqrt(df_twiss['betx'][ind_s_dead]) * part_dict['px'][dead_ind_final]
        Y_dead = part_dict['y'][dead_ind_final] / np.sqrt(df_twiss['bety'][ind_s_dead]) 
        PY_dead = df_twiss['alfy'][ind_s_dead] / np.sqrt(df_twiss['bety'][ind_s_dead]) * part_dict['y'][dead_ind_final] + np.sqrt(df_twiss['bety'][ind_s_dead]) * part_dict['py'][dead_ind_final]
        
        ### First and last turn of normalized phase space ####
        fig, ax = plt.subplots(2, 1, figsize = (8, 7.5), sharex=True, constrained_layout=True)
            
        # Final normalized X and Y
        ax[0].plot(X_alive, PX_alive, '.', 
                color='blue', markersize=3.6, label='Alive')
        ax[0].plot(X_dead, PX_dead, '.', 
                color='red', alpha=0.65, markersize=4.6, label='Killed')
        ax[0].axvline(x=x_min_norm_aperture, ls='-', color='red', alpha=0.7, label='Min. aperture')
        ax[0].axvline(x=-x_min_norm_aperture, ls='-', color='red', alpha=0.7, label=None)
        ax[0].legend(loc='upper right', fontsize=13)
        ax[0].set_ylabel('$P_{X}$')
        ax[0].set_xlabel('$X$')
        
        ax[1].plot(Y_alive, PY_alive, '.', 
                color='blue', markersize=3.6, label='Alive')
        ax[1].plot(Y_dead, PY_dead, '.', 
                color='red', alpha=0.65, markersize=4.6, label='Killed')
        ax[1].axvline(x=y_min_norm_aperture, ls='-', color='red', alpha=0.7, label='Min. aperture')
        ax[1].axvline(x=-y_min_norm_aperture, ls='-', color='red', alpha=0.7, label=None)
        #ax[1].legend(loc='upper right', fontsize=13)
        ax[1].set_ylabel('$P_{Y}$')
        ax[1].set_xlabel('$Y$')
        ax[1].text(0.01, 0.85, extra_text_string, transform=ax[1].transAxes, fontsize=9)

        # Adjust axis limits and plot turn
        ax[1].set_xlim(-0.005, 0.005)
        for a in ax:
            a.set_ylim(-0.005, 0.005)

        ### Plot physical X and Y coordinates - with colormap ###
        fig01, ax01 = plt.subplots(1, 1, figsize = (8, 7), constrained_layout=True)
        
        # Create density map of dead particles
        xy = np.vstack([part_dict['x'][dead_ind_final], part_dict['y'][dead_ind_final]]) # Calculate the point density
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()  # Sort the points by density, so that the densest points are plotted last
        x, y, z = part_dict['x'][dead_ind_final][idx], part_dict['y'][dead_ind_final][idx], z[idx]
        
        ax01.plot(part_dict['x'][alive_ind_final], part_dict['y'][alive_ind_final], '.', 
                color='blue', markersize=3.6, label='Alive')
        ax01.scatter(x, y, c=z, cmap='cool', s=2, label='Killed')
        ax01.axvline(x=x_min_aperture, ls='-', color='red', alpha=0.7, label='Min. aperture')
        ax01.axvline(x=-x_min_aperture, ls='-', color='red', alpha=0.7, label=None)
        ax01.axhline(y=y_min_aperture, ls='-', color='red', alpha=0.7, label=None)
        ax01.axhline(y=-y_min_aperture, ls='-', color='red', alpha=0.7, label=None)
        ax01.legend(loc='upper right', fontsize=13)
        ax01.set_ylabel('$y$ [m]')
        ax01.set_xlabel('$x$ [m]')
        ax01.set_ylim(-0.03, 0.03)
        ax01.set_xlim(-0.05, 0.05)
        ax01.text(0.01, 0.85, extra_text_string, transform=ax01.transAxes, fontsize=9)


        ### Plot normalized X and Y coordinates - with colormap ###
        fig1, ax1 = plt.subplots(1, 1, figsize = (8, 7), constrained_layout=True)
        
        # Create density map of dead particles
        xy2 = np.vstack([X_dead, Y_dead]) # Calculate the point density
        z2 = gaussian_kde(xy2)(xy2)
        idx2 = z2.argsort()  # Sort the points by density, so that the densest points are plotted last
        x2, y2, z2 = X_dead[idx2], Y_dead[idx2], z2[idx2]
        
        ax1.plot(X_alive, Y_alive, '.', 
                color='blue', markersize=3.6, label='Alive')
        ax1.scatter(x2, y2, c=z2, cmap='cool', s=2, label='Killed')
        ax1.axvline(x=x_min_norm_aperture, ls='-', color='red', alpha=0.7, label='Min. aperture')
        ax1.axvline(x=-x_min_norm_aperture, ls='-', color='red', alpha=0.7, label=None)
        ax1.axhline(y=y_min_norm_aperture, ls='-', color='red', alpha=0.7, label=None)
        ax1.axhline(y=-y_min_norm_aperture, ls='-', color='red', alpha=0.7, label=None)
        ax1.legend(loc='upper right', fontsize=13)
        ax1.set_ylabel('$Y$')
        ax1.set_xlabel('$X$')
        ax1.text(0.01, 0.85, extra_text_string, transform=ax1.transAxes, fontsize=9)
        ax1.set_ylim(-0.0045, 0.0045)
        ax1.set_xlim(-0.0045, 0.0045)
        
        # Print signficant losses
        bin_heights_element_where_lost, bin_borders_element_where_lost = np.histogram(part_dict['s'][dead_ind_final], bins=400)
        bin_widths_element_where_lost = np.diff(bin_borders_element_where_lost)
        bin_centers_element_where_lost = bin_borders_element_where_lost[:-1] + bin_widths_element_where_lost / 2

        s_loss, loss_count = np.unique(part_dict['s'][dead_ind_final], return_counts=True)
        ind_losses = loss_count > 200 # where more than 200 particles are lost
        s_large_losses = s_loss[ind_losses]
        large_loss_count = loss_count[ind_losses]
    
        loss_type, loss_count = np.unique(part_dict['state'][dead_ind_final], return_counts=True)
        loss_strings = 'Loss types: {}, with occurrence {}'.format(loss_type, loss_count)
        print(loss_strings)

        for jj, s in enumerate(s_large_losses):
            ele = df_twiss.iloc[np.abs(df_twiss['s'] - s).argmin()]
            loss_str = '\n{} particles lost at s = {:.3f} m, element = {}'.format(large_loss_count[jj], ele.s, ele['name'])
            loss_strings += loss_str
            print(loss_str)
        loss_strings += '\n'
        
        ## LOST PARTICLES: at which TURN ##
        fig2, ax2 = plt.subplots(1,1,figsize=(8,6), constrained_layout=True)
        bin_heights_turn_where_lost, bin_borders_turn_where_lost = np.histogram(part_dict['at_turn'][dead_ind_final], bins=30)
        bin_widths_turn_where_lost = np.diff(bin_borders_turn_where_lost)
        bin_centers_turn_where_lost = bin_borders_turn_where_lost[:-1] + bin_widths_turn_where_lost / 2
        ax2.bar(bin_centers_turn_where_lost, bin_heights_turn_where_lost, width=bin_widths_turn_where_lost, 
                alpha=0.85, color='darkred', label='Killed particles')
        ax2.set_xlim(0.0, part_dict['at_turn'][alive_ind_final][0])
        ax2.set_ylabel('Lost particle count')
        ax2.set_xlabel('Lost at turn')
        ax2.set_ylim(0.0, 1000.)
        ax2.text(0.01, 0.85, extra_text_string, transform=ax2.transAxes, fontsize=9)

        ## LOST PARTICLES: at which ELEMENT ##
        fig3, ax3 = plt.subplots(1,1,figsize=(8,6), constrained_layout=True)
        #ax3.plot(bin_centers_element_where_lost, bin_heights_element_where_lost, 
        #        alpha=0.85, marker='o', ls='None', color='darkred', label='Killed particles')
        ax3.bar(bin_centers_element_where_lost, bin_heights_element_where_lost, width=bin_widths_element_where_lost, 
                alpha=0.85, color='darkred', label='Killed particles')
        ax3.set_ylabel('Last particle count')
        ax3.set_xlabel('s [m]')
        ax3.set_ylim(0.0, 10_000.)
        ax3.set_xlim(0.0, 7000.)
        ax3.text(0.03, 0.73, extra_text_string, transform=ax3.transAxes, fontsize=10)

        return fig, fig01, fig1, fig2, fig3, loss_strings


    def plot_tracking_vs_analytical(self,
                                    analytical_tbt,
                                    tbt_dict=None, 
                                    output_folder=None,
                                    extra_str='',
                                    ylim=None,
                                    Nb_limit=None,
                                    distribution_type='gaussian',
                                    ):
        """
        Loads beam parameter data from tracking, then comparing this with anaytically propagated parameters
        
        Parameters:
        analytical_tbt : dict
            dictionary with propagated analytical parameters
        tbt_dict : dict
            dictionary containing the TBT data. If None, loads json file.
        extra_str : str
            extra string to plot
        ylim : list
            if not None, the vertical limits on emittances vertical axis
        Nb_limit : list
            if not None, will set intensity limit
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        distribution_type : str
            either 'qgaussian', 'gaussian' or 'binomial'
        """
        os.makedirs('output_plots', exist_ok=True)
        
        if tbt_dict is None:
            tbt_dict = self.load_records_dict_from_json(output_folder=output_folder)

        time_units = tbt_dict['Seconds']
            
        # Emittances and bunch intensity 
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10.2, 4.1))
        ax1.plot(time_units, tbt_dict['exn'] * 1e6, alpha=0.7, c='turquoise', lw=1.5, label='Kinetic kick')
        ax1.plot(analytical_tbt['ctime'], analytical_tbt['exn'] * 1e6, lw=2.5, label='Nagaitsev analytical')

        ax2.plot(time_units, tbt_dict['eyn'] * 1e6, alpha=0.7, c='turquoise', lw=1.5, label='Kinetic kick')
        ax2.plot(analytical_tbt['ctime'], analytical_tbt['eyn'] * 1e6, lw=2.5, label='Nagaitsev analytical')
        
        ax3.plot(time_units, tbt_dict['Nb'], alpha=0.7, lw=2.2, c='turquoise', label='Kinetic kick')

        # Find min and max emittance values - set window limits 
        #all_emit = np.concatenate((tbt_dict['exn'], tbt_dict['eyn']))
        #min_emit = 1e6 * np.min(all_emit)
        #max_emit = 1e6 * np.max(all_emit)

        for ax in (ax1, ax2, ax3):
            ax.set_xlabel('Time [s]')

        #plt.setp(ax2.get_yticklabels(), visible=False)
        ax1.text(0.04, 0.91, '{}'.format(extra_str), fontsize=15, transform=ax1.transAxes)
        ax1.set_ylabel(r'$\varepsilon_{x}^{n}$ [$\mu$m]')
        ax2.set_ylabel(r'$\varepsilon_{y}^{n}$ [$\mu$m]')
        ax3.set_ylabel(r'Ions per bunch $N_{b}$')
        ax1.legend(fontsize=11.0, loc='lower right')
        if ylim is not None:
            ax1.set_ylim(ylim[0], ylim[1])
            ax2.set_ylim(ylim[0], ylim[1])
            ax3.set_ylim(Nb_limit[0], Nb_limit[1])
        ax1.set_xlim(time_units[0] - time_units[-1] * 0.05, time_units[-1] * 1.1)
        ax2.set_xlim(time_units[0] - time_units[-1] * 0.05, time_units[-1] * 1.1)
        f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

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
                                       inj_profile_is_after_RF_spill=True,
                                       normalize_to_integral_area=False
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
        normalize_to_integral_area: bool
            whether to normalize simulated profiles to integral of measurements, rather than height. Default is False.
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
        fig0.savefig('output_plots/SPS_Zeta_Beam_Profile.png', dpi=250)
        
        #### Also generate plots comparing with profile measurements
        if also_compare_with_profile_data:
            
            # Load data, also after the RF spill - already normalized
            zeta_SPS_inj, zeta_SPS_final, zeta_PS_BSM, data_SPS_inj, data_SPS_final, data_PS_BSM = self.load_longitudinal_profile_data()
            zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill = self.load_longitudinal_profile_after_SPS_injection_RF_spill()

            # Plot longitudinal phase space, initial and final state
            fig, ax = plt.subplots(1, 2, figsize = (12, 6), sharey=True)

            ##### MEASURED PROFILES PROFILES ##### 
            ### Injection profile, after or before initial RF spill
            if inj_profile_is_after_RF_spill:
                ax[0].plot(zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill, label='SPS wall current\nmonitor data\nafter RF capture')  
            else:
                ax[0].plot(zeta_SPS_inj, data_SPS_inj, label='SPS wall current\nmonitor data at inj')  
                ax[0].plot(zeta_PS_BSM, data_PS_BSM, label='PS BSM data \nat extraction')

            #### Measured final distribution
            ax[1].plot(zeta_SPS_final, data_SPS_final, color='darkgreen', label='SPS wall current\nmonitor data\n(at ~22 s)')

            ##### SIMULATED PROFILES ##### 
            if normalize_to_integral_area and inj_profile_is_after_RF_spill:

                # Find area of measured profile until artificial ringing
                integral_limits = [tbt_dict['z_bin_centers'][0], 0.22] #tbt_dict['z_bin_centers'][-1]] # upper point in zeta is from where we 
                #integral_limits = [-np.infty, np.infty] 
                index_data = np.where((zeta_SPS_inj_after_RF_spill > integral_limits[0]) & (zeta_SPS_inj_after_RF_spill < integral_limits[1]))
                area_measured_profiles = np.trapz(data_SPS_inj_after_RF_spill[index_data], zeta_SPS_inj_after_RF_spill[index_data])

                # Find area of simulated profile until same area
                index_simulations = np.where((tbt_dict['z_bin_centers'] > integral_limits[0]) \
                                             & (tbt_dict['z_bin_centers'] < integral_limits[1]))
                
                # Find area of final, initial
                str_array = ['initial', 'final']
                color_array = ['darkturquoise', 'lime']
                for j in [0, 1]:
                    area_simulated_profiles = np.trapz(tbt_dict['z_bin_heights'][:, index_to_plot[j]][index_simulations],
                                                    tbt_dict['z_bin_centers'][index_simulations])
                    
                    area_ratio = area_measured_profiles / area_simulated_profiles
                    print('Ratio measured / simulated {}: {:4f}'.format(str_array[j], area_ratio))

                    ### Plot simulated initial distribution - normalize to same area as 
                    ax[j].plot(tbt_dict['z_bin_centers'], tbt_dict['z_bin_heights'][:, index_to_plot[j]] * area_ratio, 
                            alpha=0.8, color=color_array[j], label='Simulated {}'.format(str_array[j]))

            else:
            
                #### Plot simulated initial distribution - normalize height to 1
                ax[0].plot(tbt_dict['z_bin_centers'], tbt_dict['z_bin_heights'][:, index_to_plot[0]] / z_heights_avg[0], 
                        alpha=0.8, color='darkturquoise', label='Simulated inital')
            
                #### Simulated final distribution - normalize height to 1
                ax[1].plot(tbt_dict['z_bin_centers'], tbt_dict['z_bin_heights'][:, index_to_plot[1]] / z_heights_avg[1], 
                        alpha=0.8, color='lime', label='Simulated final')
            
 
            
            ax[0].legend(loc='upper right', fontsize=13)
            ax[1].legend(loc='upper right', fontsize=13)
            
            # Adjust axis limits and plot turn
            ax[0].set_xlim(-0.85, 0.85)
            ax[1].set_xlim(-0.85, 0.85)
            
            ax[0].text(0.02, 0.91, plot_str[0], fontsize=15, transform=ax[0].transAxes)
            ax[1].text(0.02, 0.91, plot_str[1], fontsize=15, transform=ax[1].transAxes)
            #ax[1].text(0.02, 0.85, 'Time = {:.2f} s'.format(full_data_turns_seconds_index[ind_final]), fontsize=12, transform=ax[1].transAxes)

            ax[0].set_xlabel(r'$\zeta$ [m]')    
            ax[1].set_xlabel(r'$\zeta$ [m]')
            ax[0].set_ylabel('Normalized count')
            plt.tight_layout()
            
            if inj_profile_is_after_RF_spill:
                fig.savefig('output_plots/SPS_Pb_longitudinal_profiles_vs_data_after_RF_spill.png', dpi=250)
            else:
                fig.savefig('output_plots/SPS_Pb_longitudinal_profiles_vs_data.png', dpi=250)
        plt.show()


    def plot_delta_monitor_data(self,
                                tbt_dict=None,
                                output_folder=None,
                                index_to_plot=None
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
        stack_index = np.arange(len(tbt_dict['delta_bin_heights'][0]))    
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
        delta_heights_avg = []
        for i in index_to_plot:
            # Normalize bin heights
            delta_bin_heights_sorted = np.array(sorted(tbt_dict['delta_bin_heights'][:, i], reverse=True))
            delta_height_max_avg = np.mean(delta_bin_heights_sorted[:5]) # take average of top 5 values
            delta_heights_avg.append(delta_height_max_avg)
            ax0.plot(tbt_dict['delta_bin_centers'], tbt_dict['delta_bin_heights'][:, i] / delta_height_max_avg, label=plot_str[j])
            j += 1
        ax0.set_xlabel('$\delta$ [-]')
        ax0.set_ylabel('Normalized counts')
        ax0.legend(loc='upper left', fontsize=14)
        plt.tight_layout()
        fig0.savefig('output_plots/SPS_Delta_Beam_Profile.png', dpi=250)

        # Fit q-values to delta profiles 
        output_folder_str = output_folder + '/' if output_folder is not None else ''
        n_profiles = len(tbt_dict['delta_bin_heights'][0]) 
        nturns_per_profile = tbt_dict['nturns_profile_accumulation_interval']

        sigmas_q_gaussian = np.zeros(n_profiles)
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
            with open('{}saved_delta_fits.pickle'.format(output_folder_str), 'rb') as handle:
                delta_dict = pickle.load(handle)
                sigmas_q_gaussian, q_vals, q_errors = delta_dict['sigmas_q_gaussian'], delta_dict['q_vals'], delta_dict['q_errors']
        except FileNotFoundError: 
            
            print('Fitting q-values to deltas')
            # Fit q-gaussian to delta profiles
            for i in range(n_profiles):
                delta_bin_heights_sorted = np.array(sorted(tbt_dict['delta_bin_heights'][:, i], reverse=True))
                delta_height_max_avg = np.mean(delta_bin_heights_sorted[:5]) # take average of top 5 values
                xdata, ydata = tbt_dict['delta_bin_centers'], tbt_dict['delta_bin_heights'][:, i] / delta_height_max_avg
                            
                # Fit q-Gaussian
                popt_Q, pcov_Q = fits.fit_Q_Gaussian(xdata, ydata, starting_guess_from_Gaussian=True)
                q_vals[i] = popt_Q[1]
                q_errors[i] = np.sqrt(np.diag(pcov_Q))[1] # error from covarance_matrix
                sigmas_q_gaussian[i] = fits.get_sigma_RMS_from_qGaussian_fit(popt_Q)
                print('Profile {}: q-Gaussian fit q={:.3f} +/- {:.2f}, sigma_RMS = {:.3f} m'.format(i, q_vals[i], q_errors[i], 
                                                                                                            sigmas_q_gaussian[i]))

            # Create dictionary 
            delta_dict = {'sigmas_q_gaussian': sigmas_q_gaussian, 'q_vals': q_vals, 'q_errors': q_errors, }

            # Dump saved fits in dictionary, then pickle file
            with open('{}saved_delta_fits.pickle'.format(output_folder_str), 'wb') as handle:
                pickle.dump(delta_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Dictionary with saved delta fits dumped')               
        

        ######### Plot fitted sigma_delta and q-values #########

        index_plot = np.where(sigmas_q_gaussian * 1e3 < 0.9) # otherwise unreasonable values

        f3, ax22 = plt.subplots(1, 1, figsize = (8,6))
        ax22.plot(time_array[index_plot], sigmas_q_gaussian[index_plot] * 1e3, color='cyan', ls='--', 
                  alpha=0.95, label='Simulated delta profiles')
        ax22.set_ylabel(r'$\sigma_{{\delta, RMS}}$ [m] of fitted q-gaussian [$\times 10^{3}$]')
        ax22.set_xlabel('Time [s]')
        ax22.legend()
        
        # Insert extra box with fitted m-value of profiles - plot every 10th value
        ax23 = ax22.inset_axes([0.7,0.5,0.25,0.25])
        
        # Select only reasonable q-values (above 0), then plot only every nth interval
        n_interval = 200
        q_ind = np.where((q_vals>0) & (q_vals<1.8) & (q_errors < 3.0))
        q = q_vals[q_ind]
        q_error = q_errors[q_ind]
        time_array_q = time_array[q_ind]
    
        ax23.errorbar(time_array_q[::n_interval], q[::n_interval], yerr=q_error[::n_interval], 
                        color='cyan', alpha=0.85, markerfacecolor='cyan', 
                        ls='None', marker='o', ms=5.1, label='Simulated')


        ax23.set_ylabel('Fitted $q$-value', fontsize=13.5) #, color='green')
            #ax23.legend(fontsize=11, loc='upper left')
            
        ax23.tick_params(axis="both", labelsize=12)
        #ax23.tick_params(colors='green', axis='y')
        #ax23.set_ylim(min(q)-0.2, max(q)+0.2)
        ax23.set_xlabel('Time [s]', fontsize=13.5)
        
        f3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        f3.savefig('output_plots/sigma_delta_rms_and_qvalues.png', dpi=250)
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