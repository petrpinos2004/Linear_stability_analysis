import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
import linear_stability as mother

#####################################################
# INVERSION REGIME
#####################################################

class Analysis(mother.Stability):

    def __init__(self, mode):
        
        setup = {
            'N' : 200,
            'L' : 0.3,
            'kx' : 3.117,
            'D_val' : 0.1,
            'alpha_T' : -1.7e-5,
            'alpha_p' : 0.26,
            'cp_b' : 6043,
            'a' : 0.5,
            't' : 10,
            'A_T' : 0.05,
            'f' : 0.01,
            'kappa' : 2.37e-7,
            'gamma' : 1.9,
            'delta_diff' : 0.00409, # Stokes diffusion layer thickness
            'nu' : 1.83e-7,
            'rho' : 7.3, 
            'lambda' : 0.01,
        }

        super().__init__(setup, mode)

experiment = Analysis(mode='modulation')
experiment.run()
       