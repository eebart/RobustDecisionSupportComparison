import os
import pandas as pd
import numpy as np

from ema_workbench import (Scenario, Policy)

import methodFuncs.mordm as mordm
import methodFuncs.moro as moro
import methodFuncs.pareto as pareto
import methodFuncs.reevaluate as reevaluate
import methodFuncs.robustness as robustness

import util.util as util
from util.NSGAIIHybrid import NSGAIIHybrid

from modelConfig import baseModelParams, baseModel


class BaseParams(object):
    def __init__(self, name, rootFolder, optimize=True,
                 reevaluate=True, reevaluate_scenarios=True, robust=True):
        # ----------------------------------
        # Optimization
        # ----------------------------------
        self.name = name
        self.createNewOptimizationResults = optimize
        self.numberOptimizationRepetitions = 50
        self.nfeOptimize = {
            'dps': 100000,
            'plannedadaptive': 100000,
            'intertemporal': 500000
        }
        self.algoName = 'NSGAIIHybrid'
        self.algorithm = NSGAIIHybrid
        self.epsilons = [0.1, 0.1, 0.01, 0.01]
        # max_p, utility, inertia, reliability

        self.optimizeOutputFolder = rootFolder + '/' + name + '/optimize/'

        # ----------------------------------
        # Reevaluation
        # ----------------------------------
        self.createNewReevaluationResults = reevaluate
        self.createNewReevaluationScenarios = reevaluate_scenarios
        self.numEvaluationScenarios = 10000
        self.evaluationScenarios = None
        self.baseScenario = baseModelParams.values()

        self.reevaluateOutputFolder = rootFolder + '/' + name + '/reevaluate/'

        self.createNewRobustResults = robust
        self.robustOutputFolder = rootFolder + '/' + name + '/robustness/'


class MordmParams(BaseParams):
    def __init__(self, rootFolder, optimize=True,
                 reevaluate=True, reevaluate_scenarios=True, robust=True):
        BaseParams.__init__(self, 'mordm', rootFolder, optimize,
                            reevaluate, reevaluate_scenarios, robust)

        # Optimization
        self.numberOptimizationRepetitions = 50


class MultiParams(BaseParams):
    def __init__(self, rootFolder, optimize=True,
                 reevaluate=True, reevaluate_scenarios=True, robust=True):
        BaseParams.__init__(self, 'multi', rootFolder, optimize,
                            reevaluate, reevaluate_scenarios, robust)

        # Optimization
        self.numberOptimizationRepetitions = 50
        self.references = {
            'dps': [{'b': 0.268340928, 'q': 3.502868198, 'mean': 0.042989126,
                     'delta': 0.942898012, 'stdev': 0.002705703},
                    {'b': 0.100879116, 'q': 3.699779508, 'mean': 0.045289236,
                     'delta': 0.948070276, 'stdev': 0.004367072},
                    {'b': 0.218652257, 'q': 2.050630370, 'mean': 0.042750036,
                     'delta': 0.960489157, 'stdev': 0.002456333},
                    {'b': 0.161967233, 'q': 3.868530616, 'mean': 0.038845393,
                     'delta': 0.932797521, 'stdev': 0.002184162}],
            'plannedadaptive': [{'b': 0.169003086, 'delta': 0.957020089,
                                 'mean': 0.028018711, 'q': 3.916284778,
                                 'stdev': 0.002377193},
                                {'b': 0.266894204, 'delta': 0.960659827,
                                 'mean': 0.023671571, 'q': 2.599721870,
                                 'stdev': 0.001556178},
                                {'b': 0.118192032, 'delta': 0.935636155,
                                 'mean': 0.047359169, 'q': 2.108233830,
                                 'stdev': 0.003011717},
                                {'b': 0.133371950, 'delta': 0.937291918,
                                 'mean': 0.019156371, 'q': 2.135127491,
                                 'stdev': 0.002943669}],
            'intertemporal': [{'b': 0.276034592, 'delta': 0.931005504,
                               'mean': 0.003875525, 'q': 3.049006852,
                               'stdev': 0.003875525},
                              {'b': 0.135001396, 'delta': 0.961300436,
                               'mean': 0.040694885, 'q': 2.025534937,
                               'stdev': 0.002931926},
                              {'b': 0.270388871, 'delta': 0.963128854,
                               'mean': 0.016897020, 'q': 2.478273402,
                               'stdev': 0.003902757},
                              {'b': 0.100908500, 'delta': 0.931704621,
                               'mean': 0.018744784, 'q': 3.678865354,
                               'stdev': 0.00368802}]
        }


class MoroParams(BaseParams):
    def __init__(self, rootFolder, optimize=True, optimize_scenarios=True,
                 reevaluate=True, reevaluate_scenarios=True, robust=True):
        BaseParams.__init__(self, 'moro', rootFolder, optimize,
                            reevaluate, reevaluate_scenarios, robust)

        # Optimization
        self.numberOptimizationRepetitions = 10
        self.nfeOptimize['intertemporal'] = 300000

        self.createNewOptimizationScenarios = optimize_scenarios
        self.numberOptimizationScenarios = 50
        self.optimizationScenarios = None


def outputFileEnd(model, params, refScenario=-1):
    return model.name + '_' + params.algoName + \
           '_runs' + str(params.numberOptimizationRepetitions) + \
           '_nfe' + str(params.nfeOptimize[model.name]) + \
           '_scenarios' + str(params.numEvaluationScenarios) + \
           '_refScenario' + str(refScenario)


def moea_mordm(model, params):
    return mordm.runMoea(model, params=params,
                         fileEnd=outputFileEnd(model, params),
                         reference=Scenario('reference', **baseModelParams),
                         refNum=-1)


def moea_multi(model, params):
    archives = []
    convergences = []

    refs = params.references[model.name] + [baseModelParams]
    for idx, ref in enumerate(refs):
        refScenario = (-1 if idx == len(refs)-1 else idx)
        print('Reference scenario', refScenario)
        fileEnd = outputFileEnd(model, params, refScenario=refScenario)
        results = mordm.runMoea(model, params=params,
                                fileEnd=fileEnd,
                                reference=Scenario('reference', **ref),
                                refNum=refScenario)
        archives.append(results[0])
        convergences.append(results[1])

    return(archives, convergences)


def moea_moro(model, params):
    fileEnd = outputFileEnd(model, params)
    outputFile = 'optimizeScenarios_' + fileEnd + '.csv'
    scens = moro.buildOptimizationScenarios(baseModel, params=params,
                                            outputFile=outputFile)
    params.optimizationScenarios = scens
    return moro.runMoea(model, params=params, fileEnd=fileEnd, refNum=-1)


def pareto_mordm(model, params, results=None):
    return runPareto(model=model, params=params, archive=results[0],
                     outcomes=model.outcomes, refNum=-1)


def pareto_multi(model, params, results=None):
    nondominated = []
    for idx, archive in enumerate(results[0]):
        refNum = -1 if idx == len(results[0])-1 else idx
        print('Reference scenario', refNum)

        nond = runPareto(model=model, params=params, archive=archive,
                         outcomes=model.outcomes,
                         refNum=refNum)
        nondominated.append(nond)
    return nondominated


def pareto_moro(model, params, results=None):
    return runPareto(model=model, params=params, archive=results[0],
                     outcomes=moro.robustnessFunctions, refNum=-1)


def reevaluate_one(model, params, nondominated=None):
    return runReevaluate(model=model, params=params, nondominated=nondominated,
                         fileEnd=outputFileEnd(model, params))


def reevaluate_multi(model, params, nondominated=None):
    results = []
    for idx in range(len(params.references[model.name])+1):
        ref = (-1 if idx == len(params.references[model.name]) else idx)
        print('Reference scenario', ref)

        fileEnd = outputFileEnd(model, params, refScenario=ref)
        nond = (None if nondominated is None else nondominated[idx])
        results.append(runReevaluate(model=model, params=params,
                                     nondominated=nond,
                                     fileEnd=fileEnd))
    return results


def robustness_one(model, params, results=None):
    outputFile = params.robustOutputFolder + 'robustness_' + \
                 outputFileEnd(model, params) + '.csv'
    robustData = runRobustness(model=model, params=params, results=results,
                            outputFile=outputFile)
    outputFile = (params.robustOutputFolder + 'summary_' +
                  outputFileEnd(model, params) + '.csv')
    summary = runRobustSummary(model=model, params=params,
                               robustData=robustData,
                               outputFile=outputFile)
    return (robustData, summary)


def robustness_multi(model, params, results=None):
    robusts = []
    summaries = []
    for idx in range(len(params.references[model.name])+1):
        ref = (-1 if idx == len(params.references[model.name]) else idx)
        print('Reference scenario', ref)

        result = (None if results is None else results[idx])
        outputFile = (params.robustOutputFolder + 'robustness_' +
                      outputFileEnd(model, params) +
                      '_refScenario' + str(ref) + '.csv')
        robustData = runRobustness(model=model, params=params,
                                   results=result,
                                   outputFile=outputFile)
        robusts.append(robustData)

        outputFile = (params.robustOutputFolder + 'summary_' +
                      outputFileEnd(model, params, ref) +
                      '_refScenario' + str(ref) + '.csv')
        summary = runRobustSummary(model=model, params=params,
                                   robustData=robustData,
                                   outputFile=outputFile)
        summaries.append(summary)
    return (robusts, summaries)


def robustSummary_one(model, params, robustData=None):
    outputFile = (params.robustOutputFolder + 'summary_' +
                  outputFileEnd(model, params) + '.csv')
    return runRobustSummary(model=model, params=params, robustData=robustData,
                            outputFile=outputFile)


def robustSummary_multi(model, params, robustData=None):
    summaries = []
    for idx in range(len(params.references[model.name])+1):
        ref = (-1 if idx == len(params.references[model.name]) else idx)
        data = (None if robustData is None else robustData[idx])
        outputFile = (params.robustOutputFolder + 'summary_' +
                      outputFileEnd(model, params, ref) + '.csv')
        summary = runRobustSummary(model=model, params=params,
                                   robustData=data,
                                   outputFile=outputFile)
        summaries.append(summary)
    return summaries


def scores_one(model, params, robustData=None):
    outputFile = (params.robustOutputFolder + 'scores_' +
                  outputFileEnd(model, params) + '.csv')
    return runScores(model=model, params=params, robustData=robustData[0],
                     outputFile=outputFile)


def scores_multi(model, params, robustData=None):
    scores = []
    for idx in range(len(params.references[model.name])+1):
        ref = (-1 if idx == len(params.references[model.name]) else idx)
        data = (None if robustData[0] is None else robustData[0][idx])
        outputFile = (params.robustOutputFolder + 'scores_' +
                      outputFileEnd(model, params) +
                      '_refScenario' + str(ref) + '.csv')
        score = runScores(model=model, params=params,
                          robustData=data,
                          outputFile=outputFile)
        scores.append(score)
    return scores


def runPareto(model, params, refNum=None, archive=None, outcomes=None):
    outputFile = (params.optimizeOutputFolder + 'nondominated_' +
                  outputFileEnd(model, params, refNum) + '.csv')
    if (params.createNewOptimizationResults):
        nondominated = pareto.findParetoFront(archive, outcomes=outcomes,
                                              epsilons=params.epsilons)
        print('Policies on front: ' + str(nondominated.shape[0]))
        if not os.path.exists(params.optimizeOutputFolder):
            os.makedirs(params.optimizeOutputFolder)
        nondominated.to_csv(outputFile, index=False)
    else:
        print('Loading pareto from ' + outputFile)
        nondominated = pd.read_csv(outputFile)

    return nondominated


def runReevaluate(model, params, nondominated, fileEnd):
    outputFile = 'reevaluationScenarios_' + fileEnd + '.csv'
    s = reevaluate.buildReevaluationScenarios(model=baseModel, params=params,
                                              baseScenario=baseModelParams,
                                              outputFile=outputFile)
    params.evaluationScenarios = s

    if params.createNewReevaluationResults:
        cases = nondominated[util.getLeverNames(model)].to_dict('records')
        policies = [Policy(str(i), **entry) for i, entry in enumerate(cases)]
    else:
        policies = []

    ouputFile = 'reevaluation_' + fileEnd + '.tar.gz'
    results = reevaluate.performReevaluation(model, params=params,
                                             policies=policies,
                                             outputFile=ouputFile)
    return results


def runRobustness(model, params, results, outputFile):
    if params.createNewRobustResults:
        robustData = robustness.robust_calc(results, modelName=model.name)
        if not os.path.exists(params.robustOutputFolder):
            os.makedirs(params.robustOutputFolder)
        robustData.to_csv(outputFile, index=False)
    else:
        print('Loading Robustness from ' + outputFile)
        robustData = pd.read_csv(outputFile)
    return robustData


def runRobustSummary(model, params, robustData, outputFile):
    if params.createNewRobustResults:
        toKeep = util.getLeverNames(model)
        toKeep.extend(['policy', 'model', 'max_P_percent',
                       'reliability_percent', 'utility_percent',
                       'inertia_percent'])
        summary = robustData.drop_duplicates(
            subset=['max_P_percent', 'reliability_percent',
                    'utility_percent', 'inertia_percent'],
            keep='first')
        summary = summary[toKeep]

        summary.to_csv(outputFile, index=False)
    else:
        print('Loading Robust Summary from ' + outputFile)
        summary = pd.read_csv(outputFile)
    return summary


def runScores(model, params, robustData, outputFile):
    if params.createNewRobustResults:
        result = robustness.robust_score(robustData, modelName=model.name)

        if not os.path.exists(params.robustOutputFolder):
            os.makedirs(params.robustOutputFolder)
        result.to_csv(outputFile, index=False)
    else:
        print('Loading Scores from ' + outputFile)
        result = pd.read_csv(outputFile)

    return result


methodFunctions = {
    'mordm': {
        'steps': ['moea', 'pareto', 'reevaluate', 'robust', 'score'],
        'moea': moea_mordm,
        'pareto': pareto_mordm,
        'reevaluate': reevaluate_one,
        'robust': robustness_one,
        'score': scores_one,
        'summary': robustSummary_one,
    },
    'multi': {
        'steps': ['moea', 'pareto', 'reevaluate', 'robust', 'score'],
        'moea': moea_multi,
        'pareto': pareto_multi,
        'reevaluate': reevaluate_multi,
        'robust': robustness_multi,
        'score': scores_multi,
        'summary': robustSummary_multi,
    },
    'moro': {
        'steps': ['moea', 'pareto', 'reevaluate', 'robust', 'score'],
        'moea': moea_moro,
        'pareto': pareto_moro,
        'reevaluate': reevaluate_one,
        'robust': robustness_one,
        'score': scores_one,
        'summary': robustSummary_one,
    }
}
