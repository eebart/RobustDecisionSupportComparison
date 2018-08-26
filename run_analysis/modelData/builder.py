from ema_workbench import (Model)
from ema_workbench import (RealParameter)

import modelData.dps
import modelData.intertemporal
import modelData.planned_adaptive


def base_model(params):
    base = Model('base', function=modelData.dps.lake_model)
    base.uncertainties = params.uncertainties

    return base


def dps_model(params):
    dps = Model('dps', function=modelData.dps.lake_model)
    dps.timeHorizon = params.timeHorizon
    dps.uncertainties = params.uncertainties
    dps.levers = [RealParameter("c1", -2, 2),
                  RealParameter("c2", -2, 2),
                  RealParameter("r1", 0, 2),
                  RealParameter("r2", 0, 2),
                  RealParameter("w1", 0, 1)]
    dps.outcomes = params.outcomes
    dps.constants = params.constants
    dps.constraints = params.constraints

    return dps


def planned_adaptive_model(params):
    adaptive = Model('plannedadaptive',
                     function=modelData.planned_adaptive.lake_model)
    adaptive.timeHorizon = params.timeHorizon
    adaptive.uncertainties = params.uncertainties
    adaptive.levers = [RealParameter("c1", -2, 2),
                       RealParameter("c2", -2, 2),
                       RealParameter("r1", 0, 2),
                       RealParameter("r2", 0, 2),
                       RealParameter("w1", 0, 1)]
    adaptive.outcomes = params.outcomes
    adaptive.constants = params.constants
    adaptive.constraints = params.constraints

    return adaptive


def intertemporal_model(params):
    intertemporal = Model('intertemporal',
                          function=modelData.intertemporal.lake_model)
    intertemporal.timeHorizon = params.timeHorizon
    intertemporal.uncertainties = params.uncertainties
    intertemporal.levers = [
            RealParameter('l{}'.format(i), 0, 0.1) for i in range(100)
    ]
    intertemporal.outcomes = params.outcomes
    intertemporal.constants = params.constants
    intertemporal.constraints = params.constraints

    return intertemporal
