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
            'Pr' : 1.0,
            'Ra_th' : 1e4,
            'kx' : 3.117,
            'D_val' : 0.1,
            'alpha_b' : 1.0,
            'alpha_T' : -1.0,
            'alpha_p' : 1.0,
            'cp_b' : 1.0,
            'a' : 0.5,
            't' : 10,
            'A_T' : 0.1,
            'omega' : 0.01,
            'kappa' : 1e-7,
            'mu' : 0.1,
            'gamma' : 1,
            'tau_D' : 1,
        }

        super().__init__(setup, mode)

experiment = Analysis(mode='modulation')
experiment.run()
       