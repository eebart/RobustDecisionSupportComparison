from start_run import *

methodOrder = ['mordm','multi','moro']
methodTitle = {
    'mordm': 'MORDM',
    'multi': 'Multi-Scenario MORDM',
    'moro': 'MORO'
}
modelOrder = ['intertemporal','plannedadaptive','dps']
modelTitle = {
    'dps':'Direct Policy Search\n(t=1)',
    'plannedadaptive': 'Planned Adaptive DPS\n(t=10)',
    'intertemporal': 'Intertemporal'
}
robustOutcomeOrder = ['max_P_percent', 'reliability_percent','utility_percent','inertia_percent']
robustOutcomeTitle = {
    'max_P_percent': 'Pollution',
    'reliability_percent': 'Reliability',
    'utility_percent': 'Utility',
    'inertia_percent': 'Inertia'
}

formatting = {
    'method': {
        'title': methodTitle,
        'order': methodOrder
    },
    'model': {
        'title': modelTitle,
        'order': modelOrder
    },
    'outcome': {
        'title':robustOutcomeTitle,
        'order':robustOutcomeOrder
    }
}
def titles(formatName, titleOrder=None):
    titles = []
    if titleOrder is None:
        titleOrder = order(formatName)
    for title in titleOrder:
        titles.append(formatting[formatName]['title'][title])
    return titles
def order(formatName):
    return formatting[formatName]['order']
