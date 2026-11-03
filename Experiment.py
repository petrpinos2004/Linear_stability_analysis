import matplotlib.pyplot as plt
import linear_stability as mother

#####################################################
# INVERSION REGIME
#####################################################

class Analysis(mother.Stability):

    def __init__(self):
        
        setup = {
            'N' : 200,
            'L' : 0.3,
            'kx' : 21/2,
            'a' : 1e-3,
            't' : 60,
            'dt' : 0.1,
            'A_T' : 0.05,
            'f' : 0.01,
            'T_T' : 5.1,
            'T_B' : 5,
            'p0' : 67000,
        }

        super().__init__(setup)

experiment = Analysis()
experiment.run()
       