from scipy.optimize import brentq
import numpy as np
import pandas as pd
import os

from math import floor

import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool


def critp_func(x, b, q):
    val = x ** q / (1 + x ** q) - b * x
    return val


def split(dfm, chunk_size):
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)


def buildMerged(results, scenarios=None):
    experiments, outcomes = results
    inputs = pd.DataFrame.from_records(experiments)
    outputs = pd.DataFrame.from_records(outcomes)
    merged = inputs.merge(outputs, how='outer',
                          left_index=True, right_index=True)

    if scenarios is not None:
        refs = scenarios[['reference_scenario', 'num']].reset_index(drop=True)
        refs = refs.iloc[merged['policy']].reset_index(drop=True)
        merged['reference_scenario'] = refs['reference_scenario']
        merged['num'] = refs['num']

    return merged


def calcCritP(chunk):
    printcount = floor(chunk.shape[0]/5.)
    for idx in range(chunk.shape[0]):
        critp = brentq(critp_func, 0.0000001, 1,
                       args=(chunk.iloc[idx, chunk.columns.get_loc('b')],
                             chunk.iloc[idx, chunk.columns.get_loc('q')]))
        try:
            chunk.iloc[idx, chunk.columns.get_loc('crit_p')] = critp
        except Exception:
            print(chunk.shape, idx, chunk.columns.get_loc('crit_p'), critp)
    return chunk


def robust_calc(results, modelName='', scenarios=None):
    merged = buildMerged(results, scenarios)
    merged['crit_p'] = 100

    no_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=no_workers)

    # calculate the chunk size as an integer
    chunk_size = int(merged.shape[0]/no_workers)
    chunks = [merged.ix[merged.index[i:i + chunk_size]]
              for i in range(0, merged.shape[0], chunk_size)]
    try:
        chunks[no_workers-1] = pd.concat([chunks[no_workers-1],
                                          chunks[no_workers]])
        del chunks[no_workers]
    except Exception:
        pass
    for chunk in chunks:
        chunk = chunk.copy(deep=True)
    result = pool.map(calcCritP, chunks)
    pool.close()
    pool.join()

    for i in range(len(result)):
        merged.ix[result[i].index] = result[i]

    merged['max_P_percent'] = 0
    merged['reliability_percent'] = 0
    merged['utility_percent'] = 0
    merged['inertia_percent'] = 0

    grouped = merged.groupby('policy')

    for policy, group in grouped:
        n = group.shape[0]
        merged.loc[merged['policy'] == policy, 'max_P_percent'] = (
                            len(group[group['max_P'] <= group['crit_p']]) / n)
        merged.loc[merged['policy'] == policy, 'reliability_percent'] = (
                            len(group[group['reliability'] > 0.99]) / n)
        merged.loc[merged['policy'] == policy, 'utility_percent'] = (
                            len(group[group['utility'] > 0.75]) / n)
        merged.loc[merged['policy'] == policy, 'inertia_percent'] = (
                            len(group[group['inertia'] > 0.8]) / n)

    merged.unstack()
    merged['total'] = (merged['inertia'] + merged['max_P'] +
                       merged['reliability'] + merged['utility'])

    return merged


def robust_score(critp, modelName='', scenarios=None):
    merged = critp.copy(deep=True)
    merged['max_P_good'] = 0
    merged['utility_good'] = 0
    merged['reliability_good'] = 0
    merged['inertia_good'] = 0

    merged.loc[merged['max_P'] < merged['crit_p'], 'max_P_good'] = 1
    merged.loc[merged['utility'] > 0.1, 'utility_good'] = 1
    merged.loc[merged['reliability'] > 0.99, 'reliability_good'] = 1
    merged.loc[merged['inertia'] > 0.1, 'inertia_good'] = 1

    merged['score'] = (merged['max_P_good'] + merged['utility_good'] +
                       merged['reliability_good'] + merged['inertia_good'])

    return merged
