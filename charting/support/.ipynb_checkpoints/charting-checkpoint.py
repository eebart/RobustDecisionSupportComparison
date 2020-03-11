import sys
import os
sys.path.append('../run_analysis/')
sys.path.append('../../EMAworkbench/') # or whatever your directory is

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from math import floor

from util.util import getOutcomeNames, getLeverNames
from support.load import methodParams

# from moro import robustnessFunctions

from ema_workbench.analysis import parcoords
from ema_workbench.em_framework.optimization import to_problem

from platypus.indicators import Hypervolume
from platypus import Solution

from scipy.stats import describe

from util import *

fullColors = {
    'intertemporal': {
        'mordm': ['#2E82FF','#0A6DFF','#013C92','#02285F','#0052CC'],
        'multi': ['#ffbfd5','#fc79a7','#D51865','#801440','#EA337C',],
        'moro': ['#FFC842','#FFBE23','#A87700','#704F00','#EDA700']
    },
    'dps': {
        'mordm': ['#21CEFF','#04C8FF','#006C8A','#01465A','#009AC6'],
        'multi': ['#7916FB','#5D04D3','#2D0267','#1D0242','#3A0384'],
        'moro': ['#ff6961','#FF201B','#AD0400','#690300','#D90600']
    },
    'plannedadaptive': {
        'mordm': ['#19FFAE','#00FFA5','#009560','#00613F','#00CD85'],
        'multi': ['#D884F9','#BE55E7','#7D19A5','#4C1363','#A133CC'],
        'moro': ['#FFA44D','#FF922A','#A4560C','#422204','#FF7C00']
    },
}
primaryColors = {
    'intertemporal': {
        'mordm': '#0052CC',
        'multi': '#EA337C',
        'moro': '#D89801'
    },
    'dps': {
        'mordm': '#009AC6',
        'multi': '#3A0384',
        'moro': '#D90600'
    },
    'plannedadaptive': {
        'mordm': '#00CD85',
        'multi': '#A133CC',
        'moro': '#FF7200'
    },
}
darkGray = '#808080'

def save_fig(fig, direct, name):
    if not os.path.exists(direct):
        os.makedirs(direct)

    fig.savefig('{}/{}_lowres.png'.format(direct, name), dpi=75,
                bbox_inches='tight', format='png')
    fig.savefig('{}/{}_highres.png'.format(direct, name), dpi=300,
                bbox_inches='tight', format='png')

def show_convergence_plots(method='', model='', conv=[]):
    if not isinstance(conv, list):
        conv = [conv]
    runs = len(conv[0]['run_index'].unique())

    if len(conv) == 1:
        palettes = [sns.light_palette(primaryColors[model][method],n_colors=runs)]
    else:
        palettes = [[elem] * runs for elem in fullColors[model][method]]
        sns.palplot(fullColors[model][method])

    grouped = []
    for idx, elem in enumerate(conv):
        grouped.append(elem.groupby(['run_index']))

    xs = ['nfe','nfe','nfe','nfe','nfe',None,'nfe','nfe','epsilon_progress']
    ys = ['DE','PCX','SBX','UM','UNDX',None,'epsilon_progress','hypervolume','hypervolume']

    xax = ['NFE','NFE','NFE','NFE','NFE',None,'NFE','NFE','Epsilon Progress']
    yax = ['DE','PCX','SBX','UM','UNDX',None,'Epsilon Progress','Hypervolume', 'Hypervolume']

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15), sharex=False, sharey=False)
    for idx, ax in enumerate(axes.flatten()):
        if xs[idx] == None:
            ax.set_visible(False)
            continue

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xlabel(xax[idx])
        ax.set_ylabel(yax[idx])

        for groupIdx, thegroup in enumerate(grouped):
            for name, group in thegroup:
                ax.plot(group[xs[idx]],group[ys[idx]],color=palettes[groupIdx][name])

    plt.suptitle(modelTitle[model], fontsize=18,weight='bold',y=1.02)
    plt.tight_layout()
    # save_fig(plt, direct='images/' + method, name='convergences_' + method + '_' + model + '_C.png')
    plt.show()

def show_paracord_moro(model, nondominated):
    outcomeNames = [ func.name for func in robustnessFunctions ]
    outputNames = [name.replace('fraction ','') for name in outcomeNames]
    nond = nondominated.copy(deep=True)
    nond = nond.rename(columns=dict(zip(outcomeNames, outputNames)))
    show_paracord(method='moro', model=model, outputs=outputNames,nond_combined = nond)
def show_paracord_mordm(model, nondominated):
    show_paracord(method='mordm', model=model, outputs=['max_P','utility','inertia','reliability'],nond_combined=nondominated)
def show_paracord_multi(model, nondominated):
    show_paracord(method='multi', model=model, outputs=['max_P','utility','inertia','reliability'], nond_sep=nondominated)
def show_paracord(method='', model='', outputs=[], nond_sep=[], nond_combined=None):
    if nond_combined is not None:
        sns.set_palette(sns.light_palette(primaryColors[model][method]),
                n_colors=nond_combined.shape[0])

        limits = parcoords.get_limits(nond_combined[outputs])
        limits.loc[:, ['inertia', 'reliability']] = [[0,0], [1,1]]
        limits.loc[0, ['max_P', 'utility']] = [0,0]

        axes = parcoords.ParallelAxes(limits)

        grouped = nond_combined.groupby('reference_scenario')
        for refnum, group in grouped:
            axes.plot(group[outputs], linewidth=0.5)

        axes.invert_axis('max_P')
        # plt.suptitle(methodTitle[method] + ' (Policies: ' + str(nond_combined.shape[0]) + ')', fontsize=18,weight='bold',y=1.1)

        # save_fig(plt, direct='images/' + methodName, name='parallelnondominated_all' + methodName + '_' + model + '.png')
        plt.show()
        return

    if len(nond_sep) > 0:
        sns.palplot(fullColors[model][method])
        plt.show()
    axes = None
    # for idx, nond in enumerate(nond_sep):
    # ref = str(idx+1)
    # if idx == len(nond_sep)-1: ref = 'Base'

    limits = parcoords.get_limits(nond_sep[0][outputs])
    limits.loc[:, ['inertia', 'reliability']] = [[0,0], [1,1]]
    limits.loc[0, ['max_P', 'utility']] = [0,0]
    axes = parcoords.ParallelAxes(limits)
    for idx, nond in enumerate(nond_sep):
        axes.plot(nond[outputs], color=fullColors[model][method][idx])

    axes.invert_axis('max_P')
        # plt.suptitle(methodTitle[method] + ', Reference ' + ref + ' (Policies: ' + str(nond.shape[0]) + ')', fontsize=18,weight='bold',y=1.1)
        # save_fig(plt, direct='images/' + method, name='parallelnondominated' + method + '_' + model + '_refScenario' + str(idx) + '.png')

    plt.show()

def getHypervolumeObj(method='mordm'):
    if (method == 'moro'):
        hyp = Hypervolume(minimum=[0,0,0,0], maximum=[1,1,1,1])
    elif (method == 'multi'):
        hyp = Hypervolume(minimum=[0,0,0,0], maximum=[10, 2, 1, 1])
    else:
        hyp = Hypervolume(minimum=[0,0,0,0], maximum=[2.5, 2, 1, 1])

    return hyp
def buildHypervolumeFrame(model, convergence, archive,
                          nondominated,
                          run, method, outcomes):
    dfs = []

    solutions = []
    problem = to_problem(model, 'levers')
    if outcomes == None:
        outcomes = getOutcomeNames(model)
    hyplist = []
    for index, row in nondominated[outcomes].iterrows():
        solution = Solution(problem)
        solution.objectives[:] = list(row)
        solutions.append(solution)

    hyp = getHypervolumeObj(method=method)

    for name, grp in convergence.groupby(['run_index']):
        hyplist.append(list(grp.hypervolume)[-1])
    finalhyp = hyp.calculate(solutions)
    hyplist.append(finalhyp)

    sizes = []
    for name, grp in archive.groupby(['run_index']):
        sizes.append(grp.shape[0])
    nondsize = nondominated.shape[0]
    sizes.append(nondsize)

    df = pd.DataFrame({'hypervolume': hyplist})
    df['run'] = df.index
    df['model'] = model.name
    df['baseModel'] = model.name
    df['type'] = 'individual'
    df['size'] = sizes
    dfs.append(df)

    df = pd.DataFrame({'hypervolume': [finalhyp]})
    df['run'] = run
    df['model'] = '_' + model.name
    df['baseModel'] = model.name
    df['type'] = 'final'
    df['size'] = nondsize
    dfs.append(df)

    return pd.concat(dfs)
def prepareParetoHypervolume(nondominated, archives, convergences, numberOptimizationRepetitions, method='mordm',
                             outcomes=None):
    # Compare final hypervolumes of separate sets and merged
    dfs_total = []
    for key, nondominate in nondominated.items():
        if isinstance(nondominate, list):
            dfs = []
            for idx, nond in enumerate(nondominate):
                df = buildHypervolumeFrame(model=models[key], convergence=convergences[key][idx], archive=archives[key][idx],
                                             nondominated=nond,
                                             run=numberOptimizationRepetitions, method=method, outcomes=outcomes)
                df['reference_scenario'] = idx
                if idx == len(nondominate) - 1:
                    df['reference_scenario'] = -1
                dfs.append(df)

            dfs = pd.concat(dfs)
            dfs_total.append(dfs)
        else:
            df = buildHypervolumeFrame(model=models[key], convergence=convergences[key], archive=archives[key],
                                         nondominated=nondominated[key],
                                         run=numberOptimizationRepetitions, method=method, outcomes=outcomes)
            df['reference_scenario'] = -1
            dfs_total.append(df)

    return pd.concat(dfs_total)

def paretoFrontSizeScatter(method, pareto):
    pareto['printref'] = pareto['reference_scenario'] + 1
    pareto['hue'] = np.where(pareto['type'] == 'final', '_Ref ' + pareto['printref'].map(str), 'Ref ' + pareto['printref'].map(str))
    pareto.loc[pareto['reference_scenario'] == -1 , 'hue'] = 'Base Ref'
    pareto.loc[(pareto['reference_scenario'] == -1) & (pareto['type'] == 'final'), 'hue'] = '_Base Ref'

    numRefs = len(pareto['printref'].unique())
    if numRefs == 1:
        colorder = [-1]
    else:
        colorder = [0,1,2,3,-1]
    g = sns.lmplot(x='run', y='size', row='baseModel', col='reference_scenario',
                   col_order=colorder, hue='hue', markers=list(np.tile(['.','o'], numRefs)),
                   data=pareto, sharey = False,
                   fit_reg=False, legend=False)
    g.fig.subplots_adjust(wspace=2)

    for idx, model in enumerate(modelOrder):
        data = pareto[pareto.model == model]
        if data.shape[0] == 0:
            continue
        refs = data['reference_scenario'].unique()
        if len(refs) == 1:
            sns.regplot(x='run', y='size', data=data,
                        scatter=False, color=primaryColors[model][method],
                        ax=g.axes[idx,0], ci=0)
            for i in [0,1]:
                g.axes[idx,0].collections[i].set_facecolor(primaryColors[model][method])
                g.axes[idx,0].collections[i].set_edgecolor(primaryColors[model][method])
            plt.setp(g.axes[idx,0].collections[1], sizes=[100])
            plt.setp(g.axes[idx,0].lines,linewidth=1.5)
            g.axes[idx,0].set_title(modelTitle[model].replace('\n', ' '))
        else:
            for ref, group in data.groupby('reference_scenario'):
                refStr = str(ref+1)
                if ref == -1:
                    ref = 4
                    refStr = 'Base'
                sns.regplot(x='run', y='size', data=group,
                        scatter=False, color=fullColors[model][method][ref],
                        ax=g.axes[idx,ref], ci=0)
                for i in [0,1]:
                    g.axes[idx,ref].collections[i].set_facecolor(fullColors[model][method][ref])
                    g.axes[idx,ref].collections[i].set_edgecolor(fullColors[model][method][ref])
                plt.setp(g.axes[idx,ref].collections[1], sizes=[100])
                plt.setp(g.axes[idx,ref].lines,linewidth=1.5)
                g.axes[idx,ref].set_title(modelTitle[model].replace('\n', ' ') + ': Reference ' + refStr)

    plt.tight_layout()
    return g
def paretoHypervolumeScatter(method, data, x="run",y="hypervolume",
                             xLabel="Seed Repetition",yLabel="Hypervolume",
                             shareX=True,shareY=False):
    data['printref'] = data['reference_scenario'] + 1
    data['hue'] = np.where(data['type'] == 'final', '_Ref ' + data['printref'].map(str), 'Ref ' + data['printref'].map(str))
    data.loc[data['reference_scenario'] == -1 , 'hue'] = 'Base Ref'
    data.loc[(data['reference_scenario'] == -1) & (data['type'] == 'final'), 'hue'] = '_Base Ref'

    data.loc[data['baseModel'] == 'intertemporal', 'modelTitle'] = modelTitle['intertemporal']
    data.loc[data['baseModel'] == 'dps', 'modelTitle'] = modelTitle['intertemporal']
    data.loc[data['baseModel'] == 'adaptivedirect', 'modelTitle'] = modelTitle['intertemporal']

    numRefs = len(data['printref'].unique())
    g = sns.lmplot(data=data, x='run', y='hypervolume', col='baseModel', hue='hue',
                   markers=list(np.tile(['.','o'], numRefs)),
                   fit_reg=False, legend=False)

    for idx, model in enumerate(modelOrder):

        ct = 0
        for collect in g.axes[0,idx].collections:
            if len(g.axes[0,idx].collections) == 2:
                collect.set_facecolor(primaryColors[model][method])
                collect.set_edgecolor(primaryColors[model][method])
            else:
                collect.set_facecolor(fullColors[model][method][floor(ct)])
                collect.set_edgecolor(fullColors[model][method][floor(ct)])
            if int(ct) != ct:
                plt.setp(collect, sizes=[100])
            ct += 0.5

        g.axes[0,idx].set_title(modelTitle[model].replace('\n', ' '))

        # save_fig(plt, direct='images/' + methodName, name='pareto_sizehypervolume_scatter_' + methodName + '.png')
    return g
def paretoVsHypervolume(method, data):
    data['printref'] = data['reference_scenario'] + 1
    data['hue'] = np.where(data['type'] == 'final', '_Ref ' + data['printref'].map(str), 'Ref ' + data['printref'].map(str))
    data.loc[data['reference_scenario'] == -1 , 'hue'] = 'Base Ref'
    data.loc[(data['reference_scenario'] == -1) & (data['type'] == 'final'), 'hue'] = '_Base Ref'

    data.loc[data['baseModel'] == 'intertemporal', 'modelTitle'] = modelTitle['intertemporal']
    data.loc[data['baseModel'] == 'dps', 'modelTitle'] = modelTitle['intertemporal']
    data.loc[data['baseModel'] == 'adaptivedirect', 'modelTitle'] = modelTitle['intertemporal']

    data['hue'] = np.where(data['type'] == 'final',
                           '_Ref ' + data['reference_scenario'].map(str),
                           'Ref ' + data['reference_scenario'].map(str))
    data.loc[data['reference_scenario'] == -1, 'hue'] = 'Base Ref'
    data.loc[(data['reference_scenario'] == -1) & (data['type'] == 'final'), 'hue'] = '_Base Ref'

    numRefs = len(data['printref'].unique())
    if numRefs == 1:
        colorder = [-1]
    else:
        colorder = [0,1,2,3,-1]
    g = sns.lmplot(data=data, x='size', y='hypervolume', row='baseModel', col='reference_scenario',
                   col_order=colorder, hue='hue', markers=list(np.tile(['.','o'], numRefs)),
                   sharey=False, sharex=False, fit_reg=False, legend=False)

    for idx, model in enumerate(modelOrder):
        modelData = data[data.model == model]
        if modelData.shape[0] == 0:
            continue
        refs = modelData['reference_scenario'].unique()
        if len(refs) == 1:
            for i in [0,1]:
                g.axes[idx,0].collections[i].set_facecolor(primaryColors[model][method])
                g.axes[idx,0].collections[i].set_edgecolor(primaryColors[model][method])
            plt.setp(g.axes[idx,0].collections[1], sizes=[100])
            plt.setp(g.axes[idx,0].lines,linewidth=1.5)
            g.axes[idx,0].set_title(modelTitle[model].replace('\n', ' '))
        else:
            for ref, group in data.groupby('reference_scenario'):
                refStr = str(ref+1)
                if ref == -1:
                    ref = 4
                    refStr = 'Base'
                for i in [0,1]:
                    g.axes[idx,ref].collections[i].set_facecolor(fullColors[model][method][ref])
                    g.axes[idx,ref].collections[i].set_edgecolor(fullColors[model][method][ref])
                plt.setp(g.axes[idx,ref].collections[1], sizes=[100])
                plt.setp(g.axes[idx,ref].lines,linewidth=1.5)
                g.axes[idx,ref].set_title(modelTitle[model].replace('\n', ' ') + ': Reference ' + refStr)

    plt.tight_layout()
    return g
def getParetoSizeStatistics(pareto, method):
    todescribe = pareto[pareto['type'] != 'final']
    pareto_data = []
    for name, group in todescribe.groupby(['model', 'reference_scenario']):
        dat = {'model': name[0], 'reference_scenario':name[1], 'method':method}
        hyp = list(group['hypervolume'])
        desc = describe(hyp)
        dat['mean'] = desc.mean
        dat['variance'] = desc.variance
        dat['min'] = desc.minmax[0]
        dat['max'] = desc.minmax[1]
        dat['skewness'] = desc.skewness
        dat['kurtosis'] = desc.kurtosis
        dat['combined'] = hyp[-1]

        pareto_data.append(dat)

    pareto_df = pd.DataFrame(pareto_data)
    pareto_df = pareto_df[['method','model','reference_scenario','combined','mean','variance','min','max','skewness','kurtosis']]
    pareto_df.to_csv('images/' + method + '/pareto_table_' + method + '.csv', index=False)
    return pareto_df

def showOutcomeRanges(nondominated):
    outcomes = {}
    colors = {}
    for key, nond in nondominated.items():
        for out in models[key].outcomes:
            df = list(nond[out.name])
            if not out.name in outcomes:
                outcomes[out.name] = []
            outcomes[out.name].append(df)

        colors[key] = sns.light_palette(primaryColors[key],len(outcomes))

    sns.set_style('white')
    outcomeIdx = 0
    for key, out in outcomes.items():
        print(key)
        color = []
        for modelkey, colorset in colors.items():
            color.append(colors[modelkey][outcomeIdx])
        sns.set_palette(color)
        outcomeIdx += 1

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 6))
        for a in ax:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)

        # Draw a violinplot with a narrower bandwidth than the default
        ax = sns.violinplot(data=out, bw=.2, cut=1, linewidth=1)
        ax.set(xticklabels=list(nondominated.keys()))

        save_fig(plt, direct='images/' + methodName, name='outcomeranges_' + methodName + '_outcome' + key + '.png')
        plt.show()
def showLeverRanges(nondominated):
    for modelkey, nond in nondominated.items():
        model = models[modelkey]
        levers = getLeverNames(model)
        out = nond[levers]

        sns.set_palette(sns.light_palette(primaryColors[modelkey]),
                        n_colors=len(levers))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(1.2*len(levers), 6))
        for a in ax:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)

        # Draw a violinplot with a narrower bandwidth than the default
        ax = sns.violinplot(data=out, bw=.2, cut=1, linewidth=1)

        save_fig(plt, direct='images/' + methodName, name='leverranges_' + methodName + '_model' + modelkey + '.png')
        plt.show()
