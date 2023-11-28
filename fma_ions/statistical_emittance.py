# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * 
"""
#
#  StatisticalEmittance
#   A module to calculate the transverse emittances from the beam
#   based on PyORBIT & desy-thesis-05-014
#
#   Version : 1.0.1
#   Author  : F. Asvesta
#   Contact : fasvesta .at. cern .dot. ch
"""
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

class StatisticalEmittance(object):
    """
    class for the statistical emittance estimation
    Returns:StatisticalEmittance instance
    """

    def __init__(self, particles=None, context='CPU'):
        """
        Initialization function
        Input: particles:  [xpart distribution object]
               context:    ['CPU'|'GPU']               Context to use for the calculations (default = 'CPU')
        Returns: void
        """ 
        self.coordinate_matrix=None  
        self.context = context
        if self.context=='GPU':
            try:
                import cupy as cp
            except ImportError:
                print("# StatisticalEmittance : cupy module is required to run in GPU. ")
            self.np=cp
        else:
            try:
                import numpy as np
            except ImportError:
                print("# StatisticalEmittance : numpy module is required to run in CPU. ")
            self.np=np
        if particles:
            self.set_particles(particles)
        else:
            self.beam_matrix = None
            self.beta0 = None
            self.gamma0 = None
            self.energy0 = None
            print("#StatisticalEmittance : Provide distribution in [set_particles]")
        self.dx=None
        self.coordinate_matrix_betatronic = None
        self.emitt_x = None
        self.betx = None
        self.emitt_4d = None
        self.coupling = None

    def set_particles(self,particles):
        """
        Provide distribution to calculate emittances and optics
        Input:  particles:  [xpart distribution object]
        Returns: void
        """
        if self.coordinate_matrix is not None:
            self.dx=None
            self.coordinate_matrix_betatronic = None
            self.emitt_x = None
            self.betx = None
            self.emitt_4d = None
            self.coupling = None
        context = particles._context
        ctx2np = context.nparray_from_context_array
        mask_alive = particles.state>=1
        if self.context=='CPU':
            self.coordinate_matrix = self.np.array([ctx2np(particles.x[mask_alive]),ctx2np(particles.px[mask_alive]),
                                    ctx2np(particles.y[mask_alive]),ctx2np(particles.py[mask_alive]),
                                    ctx2np(particles.zeta[mask_alive]),ctx2np(particles.delta[mask_alive])]) 
        else:
            self.coordinate_matrix = self.np.array([particles.x[mask_alive],particles.px[mask_alive],
                                    particles.y[mask_alive],particles.py[mask_alive],
                                    particles.zeta[mask_alive],particles.delta[mask_alive]])            
        self.beam_matrix = self.np.matmul(self.coordinate_matrix,self.coordinate_matrix.T)/len(particles.x[mask_alive]) 
        self.beta0 = particles.beta0[0]
        self.gamma0 = particles.gamma0[0]
        self.energy0 = particles.energy0[0]

    def correlation(self,par1,par2, betatronic=True):
        """
        Calculation of the correlation for the beam matrices
        Inputs: par1 : [0|1|2|3|4|5]
                par2 : [0|1|2|3|4|5]
                integers corresponding to coordinates (0->x), (1->px), (2->y), (3->py), (4->z), (5->dp)
                betatronic : [bool] if True the betatronic matrices are considered (default=True)
        Returns: <(a-<a>)*(b-<b>)> = <a*b> - <a>*<b>
        """
        if par1 in range(0,6) and par2 in range(0,6):
            if betatronic:
                if par1>3 or par2>3:
                    raise IOError('#StatisticalEmittance::correlation: if betatronic par1 and par2 need to be [0|1|2|3]')
                elif self.coordinate_matrix_betatronic is None:
                    self.betatronic_matrices()
                return self.beam_matrix_betatronic[par1,par2]-self.np.nanmean(self.coordinate_matrix_betatronic[par1])*self.np.nanmean(self.coordinate_matrix_betatronic[par2])
            else:
                return self.beam_matrix[par1,par2]-self.np.nanmean(self.coordinate_matrix[par1])*self.np.nanmean(self.coordinate_matrix[par2])
        else:
            raise IOError('#StatisticalEmittance::correlation: par1 and par2 need to be [0|1|2|3|4|5]')
    
    def betatronic_matrices(self):
        """
        Evaluation of the coordinates and beam matrix excluding dispersive components
        Returns: void
        """
        if self.dx is None:
            self.calculate_dispersion()

        x_betatronic=self.coordinate_matrix[0]-self.dx*self.coordinate_matrix[5]
        px_betatronic=self.coordinate_matrix[1]-self.dpx*self.coordinate_matrix[5]
        y_betatronic=self.coordinate_matrix[2]-self.dy*self.coordinate_matrix[5]
        py_betatronic=self.coordinate_matrix[3]-self.dpy*self.coordinate_matrix[5]
        self.coordinate_matrix_betatronic=self.np.array([x_betatronic,px_betatronic,y_betatronic,py_betatronic])
        self.beam_matrix_betatronic=self.beam_matrix-self.dispersion_table*self.corr5

    def calculate_dispersion(self):
        """
        Statistical dispersion evaluation
        Returns: void
        """
        self.corr5=self.correlation(5,5, betatronic=False)
        self.dx=self.correlation(0,5, betatronic=False)/self.corr5
        self.dpx=self.correlation(1,5, betatronic=False)/self.corr5
        self.dy=self.correlation(2,5, betatronic=False)/self.corr5
        self.dpy=self.correlation(3,5, betatronic=False)/self.corr5
        disp_table=self.np.array([[self.dx.tolist()],[self.dpx.tolist()],[self.dy.tolist()],[self.dpy.tolist()],[0],[0]])
        self.dispersion_table=self.np.matmul(disp_table,disp_table.T)  
    
    def calculate_emittance(self, fourD=False):
        """
        Transverse emittance evaluation 
        Returns: void
        """
        if self.emitt_x is None:
            self.x_matrix=self.np.array([[self.correlation(0,0, betatronic=True),self.correlation(0,1, betatronic=True)],[self.correlation(1,0, betatronic=True),self.correlation(1,1, betatronic=True)]])
            self.emitt_x=self.np.sqrt(abs(self.np.linalg.det(self.x_matrix)))
            self.y_matrix=self.np.array([[self.correlation(2,2, betatronic=True),self.correlation(2,3, betatronic=True)],[self.correlation(3,2, betatronic=True),self.correlation(3,3, betatronic=True)]])
            self.emitt_y=self.np.sqrt(abs(self.np.linalg.det(self.y_matrix)))
            self.z_matrix=self.np.array([[self.correlation(4,4, betatronic=False),self.correlation(4,5, betatronic=False)],[self.correlation(5,4, betatronic=False),self.corr5]])
            self.emitt_z=self.np.sqrt(abs(self.np.linalg.det(self.z_matrix)))*self.beta0*self.energy0*4*self.np.pi/299792458.0
        if fourD:
            x_y_matrix=self.np.array([[self.correlation(0,2, betatronic=True),self.correlation(0,3, betatronic=True)],[self.correlation(1,2, betatronic=True),self.correlation(1,3, betatronic=True)]])
            full_matrix=self.np.append(self.np.append(self.x_matrix,x_y_matrix,axis=1),self.np.append(x_y_matrix.T,self.y_matrix,axis=1),axis=0)
            self.emitt_4d=self.np.sqrt(self.np.linalg.det(full_matrix))

    def calculate_coupling_factor(self):
        """
        Coupling evaluation as described in doi: 10.18429/JACoW-LINAC2018-THPO118
        Returns: void
        """
        if self.emitt_4d is None:
            self.calculate_emittance(fourD=True)
        self.coupling=self.emitt_x*self.emitt_y/self.emitt_4d - 1.
    
    def calculate_twiss_functions(self):
        """
        Twiss functions evaluation 
        Returns: void
        """
        if self.emitt_x is None:
            self.calculate_emittance()
        self.betx = self.x_matrix[0,0]/self.emitt_x
        self.alfx = - self.x_matrix[0,1]/self.emitt_x
        self.gamx = self.x_matrix[1,1]/self.emitt_x
        self.bety = self.y_matrix[0,0]/self.emitt_y
        self.alfy = - self.y_matrix[0,1]/self.emitt_y
        self.gamy = self.y_matrix[1,1]/self.emitt_y

    def measure_bunch_moments(self, particles, coupling = False):
        self.set_particles(particles)
        if coupling:
            self.calculate_emittance(fourD=True)
            self.calculate_coupling_factor()
        else:
            self.calculate_emittance()
        self.calculate_twiss_functions()
        self.bunch_moments={'nemitt_x': self.emitt_x*self.beta0*self.gamma0, 'nemitt_y': self.emitt_y*self.beta0*self.gamma0, 'emitt_z': self.emitt_z,
                            'betx': self.betx, 'bety': self.bety, 
                            'alfx': self.alfx , 'alfy': self.alfy,
                            'gamx': self.gamx , 'gamy': self.gamy,
                            'dx': self.dx, 'dy': self.dy,
                            'dpx': self.dpx , 'dpy': self.dpy}
        if coupling:
            self.bunch_moments['coupling']=self.coupling
            self.bunch_moments['emitt_4d']=self.emitt_4d
        return self.bunch_moments
