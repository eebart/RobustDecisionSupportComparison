import sys
import time
import os
sys.path.append('../run_analysis/')

if os.path.exists('../../EMAworkbench/'):
    sys.path.append('../../EMAworkbench/')

from modelConfig import models
from methodConfig import MordmParams, MultiParams, MoroParams, methodFunctions

from support.util import modelOrder

rootFolder  = '../data'
methodParams = {
    'mordm': MordmParams(rootFolder, optimize=False,                           reevaluate=False, reevaluate_scenarios=False, robust=False),
    'multi': MultiParams(rootFolder, optimize=False,                           reevaluate=False, reevaluate_scenarios=False, robust=False),
    'moro':  MoroParams( rootFolder, optimize=False, optimize_scenarios=False, reevaluate=False, reevaluate_scenarios=False, robust=False)
}

def load(modelsToLoad, params, reevaluate=True, robustVals=True, scoreVals=True):
    archives = {}
    convergences = {}
    nondominated = {}
    results = {}
    robusts = {}
    scores = {}

    for model in modelOrder:
        if not modelsToLoad[model]:
            continue

        print('--------------------------------')
        print('Loading MOEA ' + model)
        archive, convergence = methodFunctions[params.name]['moea'](models[model], params=params)
        if not isinstance(archive, list):
            archive = [archive]
        if not isinstance(convergence, list):
            convergence = [convergence]
        archives[model] = archive
        convergences[model] = convergence

        print('Loading Pareto for ' + model)
        nond = methodFunctions[params.name]['pareto'](models[model], results=(archives[model], convergences[model]), params=params)
        if not isinstance(nond, list):
            nond = [nond]
        nondominated[model] = nond

        if reevaluate:
            print('Loading Reevaluations ' + model)
            results[model] = methodFunctions[params.name]['reevaluate'](models[model], params=params, nondominated=None)

        if robustVals == True:
            print('Loading Robustness ' + model)
            robusts[model] = methodFunctions[params.name]['robust'](models[model], params=params, results=None)
            if not isinstance(robusts[model], list):
                robusts[model] = [robusts[model]]
        elif robustVals == 'summary':
            print('Loading Robustness Summary ' + model)
            robusts[model] = methodFunctions[params.name]['summary'](models[model], params=params, robustData=None)
            if not isinstance(robusts[model], list):
                robusts[model] = [robusts[model]]
        if scoreVals:
            print('Loading Scores ' + model)
            scores[model] = methodFunctions[params.name]['score'](models[model], params=params, robustData=None)
            if not isinstance(scores[model], list):
                scores[model] = [scores[model]]
    print('--------------------------------')
    print('Finished loading data')

    return {'archives': archives,
            'convergences': convergences,
            'nondominated': nondominated,
            'results': results,
            'robusts': robusts,
            'scores': scores}

def loadAllData(reevaluate=False, robustVals='summary', scoreVals=False):
    modelsToLoad = {
        'dps':True,
        'plannedadaptive':True,
        'intertemporal':True
    }
    data = {
        'mordm': loadData[methodParams['mordm'].name](modelsToLoad, reevaluate=reevaluate, robustVals=robustVals, scoreVals=scoreVals),
        'multi': loadData[methodParams['multi'].name](modelsToLoad, reevaluate=reevaluate, robustVals=robustVals, scoreVals=scoreVals),
        'moro':  loadData[methodParams['moro'].name](modelsToLoad, reevaluate=reevaluate, robustVals=robustVals, scoreVals=scoreVals)
    }
    return data
def loadMordmData(modelsToLoad, reevaluate=True, robustVals=True, scoreVals=True):
    return load(modelsToLoad, methodParams['mordm'], reevaluate=reevaluate, robustVals=robustVals, scoreVals=scoreVals)
def loadMultiData(modelsToLoad, reevaluate=True, robustVals=True, scoreVals=True):
    return load(modelsToLoad, methodParams['multi'], reevaluate=reevaluate, robustVals=robustVals, scoreVals=scoreVals)
def loadMoroData(modelsToLoad, reevaluate=True, robustVals= True, scoreVals=True):
    return load(modelsToLoad, methodParams['moro'], reevaluate=reevaluate, robustVals=robustVals, scoreVals=scoreVals)

loadData = {
    'mordm':loadMordmData,
    'multi':loadMultiData,
    'moro':loadMoroData
}
