from ema_workbench import (RealParameter, ScalarOutcome, Constant, Constraint)

from modelData.builder import *

modelParams = type('obj', (object,), {
    'timeHorizon': 100,
    'uncertainties': [RealParameter('b', 0.1, 0.45),
                      RealParameter('q', 2.0, 4.5),
                      RealParameter('mean', 0.01, 0.05),
                      RealParameter('stdev', 0.001, 0.005),
                      RealParameter('delta', 0.93, 0.99)],
    'outcomes':  [ScalarOutcome('max_P', kind=ScalarOutcome.MINIMIZE),
                  ScalarOutcome('utility', kind=ScalarOutcome.MAXIMIZE),
                  ScalarOutcome('inertia', kind=ScalarOutcome.MAXIMIZE),
                  ScalarOutcome('reliability', kind=ScalarOutcome.MAXIMIZE)],
    'constants': [Constant('alpha', 0.41),
                  Constant('reps', 150)],
    'constraints': [Constraint('pollution level', outcome_names="max_P",
                               function=lambda x:max(0, x-2.5))],

})
baseModelParams = {'b': .42, 'delta': .98,
                   'mean': 0.02, 'q': 2, 'stdev': 0.0017}

baseModel = base_model(modelParams)
baseModel.name = 'base'

models = {
    'dps': dps_model(modelParams),
    'plannedadaptive': planned_adaptive_model(modelParams),
    'intertemporal': intertemporal_model(modelParams)
}
