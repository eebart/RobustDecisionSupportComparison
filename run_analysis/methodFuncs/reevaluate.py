import os

import numpy as np
import pandas as pd

from ema_workbench.em_framework.samplers import (sample_uncertainties,
                                                 DefaultDesigns)
from ema_workbench.em_framework.parameters import Scenario
from ema_workbench import (MultiprocessingEvaluator,
                           save_results, load_results)


def buildReevaluationScenarios(model, params, baseScenario, outputFile):
    if params.createNewReevaluationResults:
        if params.createNewReevaluationScenarios:
            if params.evaluationScenarios is None:
                scenarios = sample_uncertainties(model,
                                                 params.numEvaluationScenarios)
                scenarios.designs.append(tuple(baseScenario.values()))
                scenarios.n += 1

                if not os.path.exists(params.reevaluateOutputFolder):
                    os.makedirs(params.reevaluateOutputFolder)
                df = pd.DataFrame(scenarios.designs, columns=scenarios.params)
                df.to_csv(params.reevaluateOutputFolder + outputFile,
                          index=False)

                return scenarios
            else:
                return params.evaluationScenarios
        else:
            df = pd.read_csv(params.reevaluateOutputFolder + outputFile)
            designs = list(df.itertuples(index=False, name=None))
            params = [item for item in baseModel.uncertainties]
            scenarios = DefaultDesigns(designs=designs,
                                       parameters=params, n=len(designs))
            scenarios.kind = Scenario

            return scenarios
    else:
        return None


def performReevaluation(model, params, policies, outputFile):
    if (params.createNewReevaluationResults):
        with MultiprocessingEvaluator(model) as evaluator:
            scenarios = params.evaluationScenarios
            results = evaluator.perform_experiments(scenarios=scenarios,
                                                    policies=policies)
        if not os.path.exists(params.reevaluateOutputFolder):
            os.makedirs(params.reevaluateOutputFolder)
        save_results(results, params.reevaluateOutputFolder + outputFile)
    else:
        print('Loading reevaluation from ' +
              params.reevaluateOutputFolder + outputFile)
        results = load_results(params.reevaluateOutputFolder + outputFile)

    return results
