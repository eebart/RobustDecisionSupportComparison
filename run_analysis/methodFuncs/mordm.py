import os
import sys

import pandas as pd
import numpy as np

from ema_workbench import (MultiprocessingEvaluator,
                           save_results, load_results)
from ema_workbench.em_framework.optimization import (Convergence, HyperVolume,
                                                     EpsilonProgress,
                                                     ArchiveLogger)

from util.NSGAIIHybrid import (NSGAIIHybrid, OperatorProbabilities)


def runMoea(model, params, fileEnd, reference=None, refNum=None):
    archiveName = 'archives_' + fileEnd
    convergenceName = 'convergences_' + fileEnd

    if not params.createNewOptimizationResults:
        print('Loading archives from ' +
              params.optimizeOutputFolder + archiveName + '.csv')
        print('Loading convergences from ' +
              params.optimizeOutputFolder + convergenceName + '.csv')
        archives = pd.read_csv(params.optimizeOutputFolder +
                               archiveName + '.csv', index_col=0)
        convergences = pd.read_csv(params.optimizeOutputFolder +
                                   convergenceName + '.csv', index_col=0)
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
            print('Optimizing Run ', i)
            if params.name == 'mordm':
                convergences = [HyperVolume(minimum=[0, 0, 0, 0],
                                            maximum=[2.5, 2, 1, 1]),
                                EpsilonProgress()]
            else:
                convergences = [HyperVolume(minimum=[0, 0, 0, 0],
                                            maximum=[10, 2, 1, 1]),
                                EpsilonProgress()]
            if (params.algoName == 'NSGAIIHybrid'):
                for j, name in enumerate(['SBX', 'PCX', 'DE', 'UNDX', 'UM']):
                    Convergence.valid_metrics.add(name)
                    convergences.append(OperatorProbabilities(name, j))

            arch, conv = evaluator.optimize(
                algorithm=params.algorithm,
                nfe=params.nfeOptimize[model.name],
                searchover='levers',
                reference=reference,
                epsilons=params.epsilons,
                convergence=convergences,
                population_size=100)

            conv['run_index'] = i
            arch['run_index'] = i

            if refNum is not None:
                conv['reference_scenario'] = refNum
                arch['reference_scenario'] = refNum

            conv.to_csv(tmpfolder + convergenceName + '_' + str(i) + '.csv')
            arch.to_csv(tmpfolder + archiveName + '_' + str(i) + '.csv')

            convs.append(conv)
            archs.append(arch)

    archives = pd.concat(archs)
    convergences = pd.concat(convs)

    archives.to_csv(params.optimizeOutputFolder + archiveName + '.csv')
    convergences.to_csv(params.optimizeOutputFolder + convergenceName + '.csv')

    return (archives, convergences)
