'''
Created on 20 mrt. 2017

@author: sibeleker

'''
import sys
import os

if os.path.exists('../EMAworkbench/'):
    sys.path.append('../EMAworkbench/')

import numpy as np
import time
import itertools
from scipy.spatial.distance import pdist
from ema_workbench import load_results
from functools import partial
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd

from ema_workbench.util.utilities import load_results

def normalize_out_dic(outcomes):
    norm_outcomes = {}
    for ooi in outcomes.keys():
        data = outcomes[ooi]
        mx = max(data)
        mn = min(data)
        if mx == mn:
            norm_outcomes[ooi] = data - mn
        else:
            norm_outcomes[ooi] = (data - mn)/(mx-mn)
    return norm_outcomes
def calculate_distance(data, oois, scenarios=None, distance='euclidean'):
    '''data is the outcomes of exploration results,
    scenarios is a list of scenario indices (decision variables),
    oois is a list of variable names,
    distance is to choose the distance metric. options:
            bray-curtis, canberra, chebyshev, cityblock (manhattan), correlation,
            cosine, euclidian, mahalanobis, minkowski, seuclidian,
            sqeuclidian, wminkowski
    returns a list of distance values
    '''
    #make a matrix of the data n_scenarios x oois
    scenario_data = np.zeros((len(scenarios), len(oois)))
    for i, s in enumerate(scenarios):
        for j, ooi in enumerate(oois):
            scenario_data[i][j] = data[ooi][s]

    distances = pdist(scenario_data, distance)
    return distances

model = sys.argv[1]
dir = '/Users/eebart/Coursework/Year2/Thesis/Dev/ThesisWorking/deep_run/scens{}/'.format('_')
fn = model + '_selected_500experiments_10policies.tar.gz'
print(dir+fn)
try:
    results = load_results(dir+fn)
except:
    print('skipping results')
    exit(0)
exp, outcomes = results
norm_new_out = normalize_out_dic(outcomes)
oois = list(outcomes.keys())

def evaluate_diversity_single(x, data=norm_new_out, oois=oois, weight=0.5, distance='euclidean'):
    '''
    takes the outcomes and selected scenario set (decision variables),
    returns a single 'diversity' value for the scenario set.
    outcomes : outcomes dictionary of the scenario ensemble
    decision vars : indices of the scenario set
    weight : weight given to the mean in the diversity metric. If 0, only minimum; if 1, only mean
    '''
    distances = calculate_distance(data, oois, list(x), distance)
    minimum = np.min(distances)
    mean = np.mean(distances)
    diversity = (1-weight)*minimum + weight*mean

    return [diversity]

def find_maxdiverse_scenarios(combinations):
    diversity = 0.0
    solutions = []
    # ct = int(len(combinations)/10)
    for idx, sc_set in enumerate(combinations):
        # if idx % ct == 0:
        #     print('Completing index ' + str(idx), int(idx/ct))

        temp_div = evaluate_diversity_single(list(sc_set))
        if temp_div[0] > diversity:
            diversity = temp_div[0]
            solutions = []
            solutions.append(sc_set)
        elif temp_div[0] == diversity:
            solutions.append(sc_set)
    #print("found diversity ", diversity)
    return diversity, solutions

totalResults = []
def startSelection(combos, count):
    reses = []
    # no_workers = 8
    # pool = multiprocessing.Pool(processes=no_workers)

    # arrs = np.array_split(combos, 1)
    arrs = [combos]
    for idx, arr in enumerate(arrs):
        # print('working on array', idx)

        no_workers = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=no_workers)

        # with open(dir + model + '_' + str(count) + '_output_scenarioselection.txt', 'a') as file:
        start_time = time.time()
        #now, divide this data for each worker
        worker_data = np.array_split(arr, no_workers)

        result = pool.imap(find_maxdiverse_scenarios, worker_data)

        #find the max of these 8
        max_diversity = 0.0
        for r in result:
            # print("result : ", r)
            if r[0] >= max_diversity:
                max_diversity = r[0]
                solutions = []
                solutions.append(r[1])
            elif r[0] == max_diversity:
                solutions.append(r[1])

        if max_diversity < 1.35: # currently known max diversity
            print('Not saving for',max_diversity, solutions)
            continue

        with open(dir + model + '_' + str(count) + '_output_scenarioselection.txt', 'a') as file:
            print('==================Saving for {} {}====================='.format(max_diversity, solutions))
            end_time = time.time()
            file.write("Calculations took {} seconds.\n".format(end_time-start_time))
            print("Calculations took {} seconds.\n".format(end_time-start_time))
            file.write("maximum diversity and solutions: {}, {} \n\n".format(max_diversity, solutions))
            print("maximum diversity and solutions: {}, {} \n\n".format(max_diversity, solutions))

        file.close()

        pool.close()
        pool.join()

    # with open(dir + model + '_' + str(count) + '_output_scenarioselection' + '.txt', 'a') as file:
    #     max_diversity = 0.0
    #     for r in reses:
    #         print("result : ", r)
    #         if r[0] >= max_diversity:
    #             max_diversity = r[0]
    #             solutions = []
    #             solutions.append(r[1])
    #         elif r[0] == max_diversity:
    #             solutions.append(r[1])
    #
    #     totalResults.append((max_diversity, solutions))
    #
    #     file.write("maximum diversity and solutions: {}, {} \n\n".format(max_diversity, solutions))
    #     print("maximum diversity and solutions: {}, {} \n\n".format(max_diversity, solutions))

# if __name__ == "__main__":
n_scen = len(outcomes[oois[0]])
indices = range(n_scen)
set_size = 4

combos = []
count = 0
for item in itertools.combinations(indices, set_size):

    if len(combos) < 10000000:
        combos.append(item)
        continue

    # startSelection(list(combos), count)
    if (count > 179 and count < 2000) or count > 265: # 2251-2299
        print('Working on count', count)
        startSelection(list(combos), count)
    else:
        print(count)

    count += 1
    combos = []

    # if count == int(sys.argv[2]) + 50:
    #     break

if len(combos) > 0:
    print('Working on count', count+1)
    startSelection(combos, count+1)

# with open(dir + model + '_output_scenarioselection.txt', 'a') as file:
#     max_diversity = 0.0
#     for r in totalResults:
#         print("result : ", r)
#         if r[0] >= max_diversity:
#             max_diversity = r[0]
#             solutions = []
#             solutions.append(r[1])
#         elif r[0] == max_diversity:
#             solutions.append(r[1])
#
#     totalResults.append((max_diversity, solutions))
#
#     file.write("maximum diversity and solutions: {}, {} \n\n".format(max_diversity, solutions))
#     print("maximum diversity and solutions: {}, {} \n\n".format(max_diversity, solutions))
