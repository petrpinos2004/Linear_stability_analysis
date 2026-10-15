import matplotlib.pyplot as plt
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
            'alpha_T' : -1.7e-5,
            'alpha_p' : 0.26,
            'alpha_pb' : 1,
            'cp_b' : 1,
            'cp' : 6043,
            'a' : 1e-3,
            't' : [10],
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

experiment = Analysis(mode='8modulation')
experiment.run()
       