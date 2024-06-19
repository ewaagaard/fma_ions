"""
Main module containing fit functions for Gaussian, Q-Gaussian and binomial functions
"""
import numpy as np
from scipy.special import gamma as Gamma
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

class Fit_Functions:
    """
    Main class containing various fit functions and curve fitting procedures
    """
    def __init__(self):
        pass
    
    ########### FIT FUNCTION - GAUSSIAN ###########
    def Gaussian(self, x, A, mean, sigma, offset):
        """Gaussian fit of the data """
        return A * np.exp(-(x - mean)**2 / (2 * sigma**2)) + offset
    
    
    def fit_Gaussian(self, x_data, y_data, p0 = None):
        """ 
        Fit Gaussian to given X and Y data (numpy arrays)
        Custom guess p0 can be provided, otherwise generate guess
        
        Returns: parameters popt
        """
        
        # if starting guess not given, provide some qualified guess from data
        if p0 is not None: 
            initial_guess = p0
        else:
            initial_amplitude = np.max(y_data) - np.min(y_data)
            initial_mean = x_data[np.argmax(y_data)]
            initial_sigma = 1.0 # starting guess for now
            initial_offset = np.min(savgol_filter(y_data,21,2))
            
            initial_guess = (initial_amplitude, initial_mean, initial_sigma, initial_offset)
        # Try to fit a Gaussian, otherwise return array of infinity
        try:
            popt, pcov = curve_fit(self.Gaussian, x_data, y_data, p0=initial_guess)
        except (RuntimeError, ValueError):
            popt = np.infty * np.ones(len(initial_guess))
            
        return popt
    
    ########### FIT FUNCTION - Q-GAUSSIAN ###########
    def _Cq(self, q, margin=5e-4):
        """
        Normalizing constant from Eq. (2.2) in https://link.springer.com/article/10.1007/s00032-008-0087-y
        with a small margin around 1.0 for numerical stability
        """
        if q < (1 - margin):
            Cq = (2 * np.sqrt(np.pi) * Gamma(1.0/(1.0-q))) / ((3.0 - q) * np.sqrt(1.0 - q) * Gamma( (3.0-q)/(2*(1.0 -q))))   
        elif (q > (1.0 - margin) and q < (1.0 + margin)):
            Cq = np.sqrt(np.pi)
        else:
            Cq = (np.sqrt(np.pi) * Gamma((3.0-q)/(2*(q-1.0)))) / (np.sqrt(q-1.0) * Gamma(1.0/(q-1.0)))
        if q > 3.0:
            raise ValueError("q must be smaller than 3!")
        else:
            return Cq
    
    def _eq(self, x, q):
        """ Q-exponential function
            Available at https://link.springer.com/article/10.1007/s00032-008-0087-y
        """
        eq = np.zeros(len(x))
        for i, xx in enumerate(x):
            if ((q != 1) and (1 + (1 - q) * xx) > 0):
                eq[i] = (1 + (1 - q) * xx)**(1 / (1 - q))
            elif q==1:
                eq[i] = np.exp(xx)
            else:
                eq[i] = 0
        return eq
    
    
    def Q_Gaussian(self, x, mu, q, beta, A, C):
        """
        Returns Q-Gaussian from Eq. (2.1) in (Umarov, Tsallis, Steinberg, 2008) 
        available at https://link.springer.com/article/10.1007/s00032-008-0087-y
        """
        Gq =  A * np.sqrt(beta) / self._Cq(q) * self._eq(-beta*(x - mu)**2, q) + C
        return Gq
    
    
    def fit_Q_Gaussian(self, x_data, y_data, q0 = 1.4):
        """
        Fits Q-Gaussian to x- and y-data (numpy arrays)
        Parameters: q0 (starting guess)
        
        Returns fitted parameters poptq and fit errors poptqe
        """
    
        # Test Gaussian fit for the first guess
        popt = self.fit_Gaussian(x_data, y_data) # gives A, mu, sigma, offset
        p0 = [popt[1], q0, 1/popt[2]**2/(5-3*q0), 2*popt[0], popt[3]] # mu, q, beta, A, offset
    
        #min_bounds = (-np.inf, -np.inf, 0.0, -np.inf, -np.inf,-np.inf)
        #max_bounds = (np.inf, 3.0, np.inf, np.inf, np.inf, np.inf)
    
        try:
            poptq, pcovq = curve_fit(self.Q_Gaussian, x_data, y_data, p0)
            poptqe = np.sqrt(np.diag(pcovq))
        except (RuntimeError, ValueError):
            poptq = np.nan * np.ones(len(p0))
            
        return poptq
    

    def Binomial(self, x, A, m, x_max, x0, offset):
        """Binomial phase space distribution"""
        #x_max = np.sqrt(sigma)
        return A * (1 - ((x-x0)/x_max)**2)**(m-0.5) + offset
    
    def fit_Binomial(self, x_data, y_data, p0=None):
        
        if p0 is None:
            p0 = [1.0, 5.3, 3.0, 0.0, 0.0]
        
        popt_B, _ = curve_fit(self.Binomial, x_data, y_data, p0)
        return popt_B