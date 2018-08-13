import numpy as np
import pandas as pd


def objToDict(theobj):
    thedict = {}
    for key, val in vars(mordmParams).items():
        if key.find('__') != -1:
            continue
        if not (isinstance(val, str) or
                isinstance(val, bool) or
                isinstance(val, int)):
            continue
        thedict[key] = [val]
    return thedict


def getLeverNames(model):
    return [item.name for item in model.levers]


def getOutcomeNames(model):
    return [item.name for item in model.outcomes]


def getUncertaintyNames(model):
    return [item.name for item in model.uncertainties]
