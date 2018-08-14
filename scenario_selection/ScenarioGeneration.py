'''
Created on Jan 17, 2018

@author: eker
'''
import numpy as np
#THE FUNCTION FOR ANTHROPOGENIC POLLUTION FOR CLOSED LOOP CONTROL
def a_t(X, #x is a scalar, pollution at time t
        c=[],
        r=[],
        w=[],
        n=2):

    a = sum([w[j]*(abs((X-c[j])/r[j]))**3 for j in range(n)])
    return min(max(a, 0.01), 0.1)

def lake_problem_closedloop(
         b = 0.42,          # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,           # recycling exponent
         mean = 0.02,       # mean of natural inflows
         stdev = 0.001,     # future utility discount rate
         delta = 0.98,      # standard deviation of natural inflows
         
         alpha = 0.4,       # utility from pollution
         nsamples = 100,    # Monte Carlo sampling of natural inflows
         timehorizon = 100, # simulation time
         **kwargs):         
    '''
    in the closed loop version, utility and inertia are included in the Monte Carlo simulations, too,
    since they are now dependent on a[t].
    '''
    print("kwargs : ", kwargs)
    c1 = kwargs['c1']
    c2 = kwargs['c2']
    r1 = kwargs['r1']
    r2 = kwargs['r2']
    w1 = kwargs['w1']
    w2 = 1 - w1

    
    Pcrit = brentq(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = int(timehorizon)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.zeros((nvars,))

    reliability = 0.0
    utility = 0.0
    inertia = 0.0

    for _ in range(nsamples):
        X[0] = 0.0
        decisions[0] = 0.0
        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
  
        for t in range(1,nvars):
            decisions[t] = a_t(X[t-1],c=[c1,c2], r=[r1,r2], w=[w1,w2])
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) +\
                    decisions[t-1] +\
                    natural_inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)
        
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)
        utility += (np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))) / float(nsamples)
        inertia += (np.sum(np.absolute(np.diff(decisions)) > 0.02)/float(nvars-1)) / float(nsamples)

      
    max_P = np.max(average_daily_P)
    return max_P, utility, inertia, reliability

import numpy as np
from ema_workbench import (Model, RealParameter, ScalarOutcome, Constant)

#instantiate the model
lake_model = Model('lakeproblem', function=lake_problem_closedloop)
lake_model.time_horizon = 100
#specify uncertainties
lake_model.uncertainties = [RealParameter('b', 0.1, 0.45),
                            RealParameter('q', 2.0, 4.5),
                            RealParameter('mean', 0.01, 0.05),
                            RealParameter('stdev', 0.001, 0.005),
                            RealParameter('delta', 0.93, 0.99)]

# set levers, one for each time step
lake_model.levers = [RealParameter("c1", 0, 0.1),
                     RealParameter("c2", 0, 0.1),
                     RealParameter("r1", 0, 0.1), #[0,2]
                     RealParameter("r2", 0, 0.1), #
                     RealParameter("w1", 0, 0.1)
                     ]

#specify outcomes 
lake_model.outcomes = [ScalarOutcome('max_P', kind=ScalarOutcome.MINIMIZE),
                       ScalarOutcome('utility', kind=ScalarOutcome.MAXIMIZE),
                       ScalarOutcome('inertia', kind=ScalarOutcome.MINIMIZE),
                       ScalarOutcome('reliability', kind=ScalarOutcome.MAXIMIZE)]

# override some of the defaults of the model
lake_model.constants = [Constant('alpha', 0.41),
                        Constant('nsamples', 100),
                        Constant('timehorizon', lake_model.time_horizon),
                       ]

import os

from ema_workbench import (perform_experiments, ema_logging, save_results, 
                           load_results, Policy)
from ema_workbench.em_framework import samplers

# turn on logging
ema_logging.log_to_stderr(ema_logging.INFO)

# perform experiments
nr_experiments = 2
nr_policies = 1
fn = './data/{}_experiments_openloop_noApollution.tar.gz'.format(nr_experiments)
policy = Policy({'c1':0,
                'c2':0,
                'r1':1,
                'r2':0,
                'w1':0})
results = perform_experiments(lake_model, 10, policies=[policy])
#try:
    # why regenerate the data?
#    results = load_results(fn)
#except IOError:
#    results = perform_experiments(lake_model, scenarios=nr_experiments)
save_results(results, fn)