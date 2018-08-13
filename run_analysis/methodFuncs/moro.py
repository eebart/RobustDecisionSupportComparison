import os
import sys

from ema_workbench import (Model, RealParameter, ScalarOutcome,
                           Constant, ReplicatorModel, Constraint,
                           Scenario, MultiprocessingEvaluator)
from ema_workbench.em_framework.samplers import (sample_uncertainties,
                                                 DefaultDesigns)
from ema_workbench.em_framework.optimization import (Convergence, HyperVolume,
                                                     EpsilonProgress,
                                                     ArchiveLogger)

from platypus import (NSGAII, EpsNSGAII)
from util.NSGAIIHybrid import (NSGAIIHybrid, OperatorProbabilities)

import pandas as pd
import numpy as np
import functools
import pickle


def getLeverNames(model):
    return [item.name for item in model.levers]


def getOutcomeNames(model):
    return [item.name for item in model.outcomes]


def countRobust(robustGoal, robustThreshold, outcomes):
    if robustGoal == 'min':
        return np.sum(outcomes <= robustThreshold) / outcomes.shape[0]
    else:
        return np.sum(outcomes >= robustThreshold) / outcomes.shape[0]


maxp = functools.partial(countRobust, 'min', 0.75)
reliability = functools.partial(countRobust, 'max', 0.99)
utility = functools.partial(countRobust, 'max', 0.75)
inertia = functools.partial(countRobust, 'max', 0.8)

robustnessFunctions = [ScalarOutcome('fraction max_P',
                                     kind=ScalarOutcome.MAXIMIZE,
                                     variable_name='max_P',
                                     function=maxp),
                       ScalarOutcome('fraction reliability',
                                     kind=ScalarOutcome.MAXIMIZE,
                                     variable_name='reliability',
                                     function=reliability),
                       ScalarOutcome('fraction inertia',
                                     kind=ScalarOutcome.MAXIMIZE,
                                     variable_name='inertia',
                                     function=inertia),
                       ScalarOutcome('fraction utility',
                                     kind=ScalarOutcome.MAXIMIZE,
                                     variable_name='utility',
                                     function=utility)]


def buildOptimizationScenarios(model, params, outputFile):
    if params.createNewOptimizationResults:
        if params.createNewOptimizationScenarios:
            if params.optimizationScenarios is None:
                scenarios = sample_uncertainties(model,
                                                 params.numEvaluationScenarios)

                if not os.path.exists(params.optimizeOutputFolder):
                    os.makedirs(params.optimizeOutputFolder)
                df = pd.DataFrame(scenarios.designs, columns=scenarios.params)
                df.to_csv(params.optimizeOutputFolder + outputFile,
                          index=False)

                return scenarios
            else:
                return params.optimizationScenarios
        else:
            df = pd.read_csv(params.optimizeOutputFolder + outputFile)
            designs = list(df.itertuples(index=False, name=None))
            scenarios = DefaultDesigns(designs=designs,
                                       parameters=model.uncertainties,
                                       n=len(designs))
            scenarios.kind = Scenario

            return scenarios
    else:
        return None


def runMoea(model, params, fileEnd, refNum=-1):
    archiveName = 'archives_' + fileEnd
    convergenceName = 'convergences_' + fileEnd

    if not params.createNewOptimizationResults:
        print('Loading archives from ' +
              params.optimizeOutputFolder + archiveName + '.csv')
        print('Loading convergences from ' +
              params.optimizeOutputFolder + convergenceName + '.csv')
        archives = pd.read_csv(params.optimizeOutputFolder
                               + archiveName + '.csv', index_col=0)
        convergences = pd.read_csv(params.optimizeOutputFolder
                                   + convergenceName + '.csv', index_col=0)
        return (archives, convergences)

    archs = []
    convs = []

    if not os.path.exists(params.optimizeOutputFolder):
        os.makedirs(params.optimizeOutputFolder)
    tmpfolder = params.optimizeOutputFolder + model.name + '/'
    if not os.path.exists(tmpfolder):
        os.makedirs(tmpfolder)

    with MultiprocessingEvaluator(model) as evaluator:
        for i in range(params.numberOptimizationRepetitions):
            print('Run ', i)
            convergence_metrics = [HyperVolume(minimum=[0, 0, 0, 0],
                                               maximum=[1, 1, 1, 1]),
                                   EpsilonProgress()]
            if (params.algoName == 'NSGAIIHybrid'):
                for j, name in enumerate(['SBX', 'PCX', 'DE', 'UNDX', 'UM']):
                    Convergence.valid_metrics.add(name)
                    convergence_metrics.append(OperatorProbabilities(name, j))

            arch, conv = evaluator.robust_optimize(
                                    robustnessFunctions,
                                    params.optimizationScenarios,
                                    algorithm=params.algorithm,
                                    nfe=params.nfeOptimize[model.name],
                                    constraints=[],
                                    epsilons=params.epsilons,
                                    convergence=convergence_metrics)

            arch['run_index'] = i
            conv['run_index'] = i
            archs.append(arch)
            convs.append(conv)

            arch.to_csv(tmpfolder + archiveName + '_' + str(i) + '.csv')
            conv.to_csv(tmpfolder + convergenceName + '_' + str(i) + '.csv')

    archives = pd.concat(archs)
    convergences = pd.concat(convs)

    archives.to_csv(params.optimizeOutputFolder + archiveName + '.csv')
    convergences.to_csv(params.optimizeOutputFolder + convergenceName + '.csv')

    return (archives, convergences)
