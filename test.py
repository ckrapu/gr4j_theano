import unittest
import numpy as np
import theano
import gr4j_theano as gt
import theano.tensor as tt
import pandas as pd
theano.config.compute_test_value = "ignore"

class TestCases(unittest.TestCase):

    def test_forward_simulate(self):


        P = tt.vector()
        E = tt.vector()

        x1 = tt.scalar()
        x2 = tt.scalar()
        x3 = tt.scalar()
        x4 = tt.scalar()
        x4_limit = tt.scalar()

        S0  = tt.scalar()
        R0  = tt.scalar()
        Pr0 = tt.vector()

        streamflow = gt.simulate_streamflow(P,E,
                          S0,Pr0,R0,
                          x1,x2,x3,x4,x4_limit)

        simulate = theano.function(inputs = [P,E,
                                     S0,Pr0,R0,
                                     x1,x2,x3,x4,x4_limit],
                           outputs = streamflow)

        data = pd.read_csv('./data/sample.csv')
        Q = simulate(data['P'].values,data['ET'].values,0.6*320.11,
        np.zeros(9),0.7 * 69.63,320.11,2.42,69.63,1.39,5)
        mae = np.mean(np.abs(Q - data['modeled_Q'].values))

        self.assertTrue(mae < 0.01)
