import numpy as np
import pandas as pd

import util.pareto as pareto


def findParetoFront(archive, outcomes=None, epsilons=None):
    archive['repetition'] = archive.index
    archive = archive.reset_index(drop=True)

    objectives = []
    maximize = []
    cols = list(archive.columns)
    for outcome in outcomes:
        objectives.append(cols.index(outcome.name))
        if outcome.kind == 1:
            maximize.append(cols.index(outcome.name))

    nondominated = pareto.eps_sort([list(archive.itertuples(False))],
                                   objectives=objectives,
                                   epsilons=epsilons,
                                   maximize=maximize)
    return pd.DataFrame(nondominated, columns=archive.columns)
