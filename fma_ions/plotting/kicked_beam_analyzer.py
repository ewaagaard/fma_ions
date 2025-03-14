"""
Main container for analyzing and plotting kicked TBT data
"""
from dataclasses import dataclass
from pathlib import Path
import os, json, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import PyNAFF as pnf

from .sps_plotting_classes import SPS_Plotting

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 16.5,
        "xtick.labelsize": 15.5,
        "ytick.labelsize": 15.5,
        "legend.fontsize": 15,
        "figure.titlesize": 20,
    }
)

@dataclass
class SPS_Kick_Plotter:
    """ 
    Plotting class object for kicked particle beams in the SPS
    """

    def get_tune_pnaf(self, data, turns=40):
        """
        Calculate tune using PyNAFF algorithm.
        
        Args:
            data: Turn-by-turn position data
            turns: Number of turns to analyze
        
        Returns:
            Tune value (frequency)
        """
        # Subtract the mean to remove the DC component
        data = data - np.mean(data)
        result = pnf.naff(data, turns=turns, nterms=1, skipTurns=0, getFullSpectrum=False, window=1)
        try:
            result = pnf.naff(data, turns=turns, nterms=1, skipTurns=0, getFullSpectrum=False, window=1)
            return result[0][1]  # Return the frequency (tune)
        except Exception as e:
            print(f"Error in PyNAFF: {e}")
            return np.nan


    def knobs_to_tunes(self, kqd: np.ndarray, kqf: np.ndarray, optics: str ='q26'):
        """Convert knob values to tunes using the inverse of the given relations."""
        if optics == 'q20':
            qx0                 =                        20.13
            qy0                 =                        20.18
            kqf0                =          0.01157926643000353
            kqd0                =         -0.01158101412515668
            qh_setvalue         =                        20.13
            qv_setvalue         =                        20.18
            dkqf_h              =        0.0003910514166916948
            dkqd_h              =       -0.0001167644785878596
            dkqf_v              =        0.0001171072522694149
            dkqd_v              =       -0.0003918448750990765
        elif optics == 'q26':
            qx0                 =                        26.13
            qy0                 =                        26.18
            kqf0                =          0.01443572827710525
            kqd0                =         -0.01443942830604748
            qh_setvalue         =                        26.13
            qv_setvalue         =                        26.18
            dkqf_h              =        0.0003701631351700831
            dkqd_h              =       -7.131174595960521e-05
            dkqf_v              =        7.150314689878521e-05
            dkqd_v              =       -0.0003705414501690567
        else:
            raise ValueError(f"Optics {optics} not supported")

        qv_setvalue = ( (kqf - kqf0 + qx0*dkqf_h + qy0*dkqf_v) * dkqd_h - (kqd - kqd0 + qx0*dkqd_h + qy0*dkqd_v) * dkqf_h ) / (dkqf_v * dkqd_h - dkqd_v * dkqf_h)
        qh_setvalue = ( (kqd - kqd0 + qx0*dkqd_h + qy0*dkqd_v) - qv_setvalue * dkqd_v ) / dkqd_h
        
        return qh_setvalue, qv_setvalue
    
    
    def plot_tbt_data_to_spectrum(self, tbt_dict=None, output_folder = 'output_tbt', t4s = 40, i_start = 200,
                             Q_int = 26, ripple_freqs=None, ion_type='Pb', transfer_function_bounds = [10., 1500.]):
        """
        Convert turn-by-turn data to normalized FFT spectrum
        
        Parameters:
        ----------
        tbt_dict : dict
            TBT data dictionary. If not given, will scan for the file
        output_folder : str 
            folder location where data is stored
        t4s : int
            number of turns for PyNAFF analysis
        i_start : int
            first turn after the kick
        Q_int : int
            integer tune
        ripple_freqs : np.ndarray
            array with excited frequencies, if given
        ion_type : str
            'Pb' or 'proton'
        transfer_function_bounds : list
            upper and lower limit to analyze for transfer function
        """
        # Extra path lengths from kicks
        extra_time = 0.0 #0.55e-6 # to account for extra path length
        extra_time_str = '_extra_time' if extra_time != 0.0 else ''
        
        # Define revolution period
        if ion_type=='Pb':
            T = 2.327274044418234e-05 + extra_time
        elif ion_type=='proton':
            T = 2.3069302183004387e-05 # 23.03e-6 + 0.06e-6  # old SPS revolution period from Hannes
        else:
            raise ValueError('Ion type needs to be either Pb or proton')
    
        ### Load data ###
        if tbt_dict is None:
            sps_plot = SPS_Plotting()
            tbt_dict = sps_plot.load_records_dict_from_json(output_folder=output_folder)
        
        # Check if processed data exists
        processed_data_file = f'{output_folder}/processed_tunes.json'
        if os.path.exists(processed_data_file):
            # Load pre-processed data
            with open(processed_data_file, 'r') as f:
                tunes = json.load(f)
            print("Loaded pre-processed tune data")
            
            # Get index for plotting
            i_stop = len(tbt_dict['X_data'])
            ind = []
            for i in range(i_start, i_stop-t4s):
                ind.append(i)
        else:            
            # Process each plane
            data = {'H': tbt_dict['X_data'], 'V': tbt_dict['Y_data']}
            tunes = {}
        
            for plane in data.keys():
                print(f'\nPlane: {plane}\n')
                delta = data[plane]
                i_stop = len(delta)
                
                # Analyze tunes
                tunes_singlebpm = []
                ind = []
                for i in range(i_start, i_stop-t4s):
                    ind.append(i)
                    if i % 1000 == 0:
                        print(f'Nr: {i}')
                    # Get tune using PyNAFF
                    d = delta[i:i+t4s]
                    tune = self.get_tune_pnaf(d, turns=t4s)
                    tunes_singlebpm.append(tune)
                
                tunes[plane] = np.array(tunes_singlebpm).tolist()  # Convert to list for JSON serialization
            
            # Save processed data
            os.makedirs(output_folder, exist_ok=True)
            with open(processed_data_file, 'w') as f:
                json.dump(tunes, f)
            print("Processed and saved tune data")
        
        # Compute tune values from knobs used in simulations
        Qx_knobs, Qy_knobs = self.knobs_to_tunes(tbt_dict['kqd_data'], tbt_dict['kqf_data'])
        
        # Convert loaded data back to numpy arrays if needed
        tunes = {plane: np.array(tune_data) for plane, tune_data in tunes.items()}
        tunes_knob = {'Qx_knobs': Qx_knobs, 'Qy_knobs': Qy_knobs}
        
        # Setup plot style
        colors = {'H': 'b', 'V': 'g'}
        colors2 = ['cyan', 'lime']
        fig, (ax_tune, ax_spectrum_H, ax_spectrum_V) = plt.subplots(3, 1, figsize=(12,11), constrained_layout=True)
        ax_spectrum = {
            'H': ax_spectrum_H,
            'V': ax_spectrum_V
        }
        
        #### TBT data ####
        fig0, ax0 = plt.subplots(2, 1, sharex=True, constrained_layout=True)
        ax0[0].plot(tbt_dict['X_data'], color='cyan')
        ax0[1].plot(tbt_dict['Y_data'], color='darkorange')
        ax0[0].set_ylabel('X [m]')
        ax0[1].set_ylabel('Y [m]')
        ax0[1].set_xlabel('Turns')
        
        # Add inset box plot of the first few 100 turns
        plane_str = ['X', 'Y']
        colors0 = ['cyan', 'darkorange']
        for i, a in enumerate(ax0):
            ax_inset = a.inset_axes([0.65, 0.48, 0.3, 0.32])
            ax_inset.plot(tbt_dict[f'{plane_str[i]}_data'][:100], color=colors0[i])
            ax_inset.tick_params(labelsize=6)

        fig0.savefig(f'{output_folder}/TBT_data.png', dpi=400)
        
        #### FFT from TBT data ####
        planes = ['H', 'V']
        tbt_spectrum = {}
        transfer_function = {}
        for i, plane in enumerate(planes):
            # Plot tune evolution
            turns_tbt = np.arange(len(tunes[plane]))
            ax_tune.plot(turns_tbt + t4s, Q_int + tunes[plane], ls='--',
                        label=f'{plane} tune from TBT', color=colors[plane])
            
            # Calculate and plot FFT of TBT tune evolution
            N = len(tunes[plane])
            yf = np.abs(fftshift(fft(tunes[plane] - np.nanmean(tunes[plane]), N))) / N
            xf = fftshift(fftfreq(N, T))
                        
            ax_spectrum[plane].semilogy(xf, yf, color=colors[plane], label='TBT tune spectrum')
            ax_spectrum[plane].set_ylabel(f'{plane}: norm. FFT amplitude')
            ax_spectrum[plane].set_xlim(0, 1500)
            ax_spectrum[plane].set_ylim(1e-7, 1e-1)
            ax_spectrum[plane].grid(True)
            
            tbt_spectrum[plane] = yf
        
                
        ### FFT from current spectrum ### 
        for i, key in enumerate(tunes_knob):
            ax_tune.plot(turns_tbt+t4s/2, tunes_knob[key][ind], 
                        label=f'{planes[i]} tune from knobs k', alpha=0.85, color=colors2[i])
            N_knob = len(tunes_knob[key][ind])
            yf_knob = np.abs(fftshift(fft(tunes_knob[key][ind] - np.nanmean(tunes_knob[key][ind]), N_knob))) / N_knob
            xf_knob = fftshift(fftfreq(N_knob, T))
            ax_spectrum[planes[i]].semilogy(xf_knob, yf_knob, ls='--', alpha=0.85, color=colors2[i], label='Knobs k tune spectrum')
            ax_spectrum[plane].legend(fontsize=13)
            
            # Add markers at 50 Hz intervals
            if ripple_freqs is not None:
                marker_indices = [np.argmin(np.abs(xf - f)) for f in ripple_freqs]
                ax_spectrum[planes[i]].plot(xf[marker_indices], 1.0/N * np.abs(yf)[marker_indices], 
                                'r.', markersize=8, label='50 Hz intervals')
                
            # Compute and append transfer function
            print('Transfer function: TBT spectrum {} divided by current {}'.format(planes[i], key))
            transfer_function[planes[i]] = tbt_spectrum[planes[i]] / yf_knob
        
        # Finalize plots
        ax_tune.set_title('Tune evolution TBT vs knobs data')
        ax_tune.set_xlabel('Turn')
        ax_tune.set_ylabel('Tune')
        ax_tune.legend(fontsize=13)
        ax_tune.grid(True)
        
        ax_spectrum_V.set_xlabel('Frequency [Hz]')
        fig.savefig(f'{output_folder}/Tune_spectrum_TBT_knobs{extra_time_str}.png', dpi=400)
        
        ### Transfer function plot ###
        ind_t = np.where((xf_knob > transfer_function_bounds[0]) & (xf_knob < transfer_function_bounds[1]))
        
        fig2, ax2 = plt.subplots(2,1,figsize=(8,6), sharex=True, constrained_layout=True)
        ax2[0].plot(xf_knob[ind_t], transfer_function['H'][ind_t])
        #ax[0].plot(frequencies_tf, fitted_transfer_function_combined, label=None, linestyle='-', color='cyan')
        ax2[1].plot(xf_knob[ind_t], transfer_function['V'][ind_t])
        #ax[1].plot(frequencies_tf, fitted_transfer_function_combined, label='Combined LC filter + C section', linestyle='-', color='cyan')
        
        # --- Add red dots for specific frequencies on the combined fit ---
        #specific_frequencies_for_plot = np.array(specific_frequencies) # Convert to numpy array for plotting
        #values_for_plot = np.array(list(combined_function_values_at_freqs.values())) # Get corresponding function values
        #ax[1].loglog(specific_frequencies_for_plot, values_for_plot, 'ro', label='Combined Fit Values') # Red dots
            
        ax2[1].set_xlabel('Frequency [Hz]')
        ax2[0].set_ylabel('X Transfer function [a.u.]', fontsize=14)
        ax2[1].set_ylabel('Y Transfer function [a.u.]', fontsize=14)
        #ax2[1].legend(fontsize=11)
        for a in ax2:
            a.set_yscale('log')
            a.set_xscale('log')
            a.grid(alpha=0.55)
            #a.set_xlim(10, 1500)
        fig2.savefig(f'{output_folder}/Transfer_functions_TBT_knobs{extra_time_str}.png', dpi=400)
        
        plt.show()