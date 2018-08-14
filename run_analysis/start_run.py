"""
Copyright (C) 2018 Erin Bartholoemw.

This and the corresponding scripts required to run it are free software.
You can redistribute or modify it at will. This script provides a strucutre
for generating the data required to compare multiple methods of robust
decision support. Use of this script was first developed to support the
research in the thesis Robust Decision Support Methods: A Comparative
Analysis by Erin Bartholomew and found in the TU Delft Repository:
<https://repository.tudelft.nl/islandora/search/?collection=education>.

===========================================================

start_run.py

Provides a strucutre for generating data for multiple robust decision support
methods in a consistent manner. This script supports three methods and three
model alternatives.

Using the methodParams dictionary, you can control which components of each
method should generate new results and which can use results from the file
system (the location is basd on the rootFolder). By changing the listed flags
to False, that section of the method will be loaded from the file system. In
this way, you can run the analysis in chunks and will not have to re-run
early steps to generate data in the later steps.

Any method configuration can be found in methodConfig.py.
Model configuration can be found in modelConfig.py, with corresponding model
code in the models folder.

"""

from modelConfig import models, baseModel
from methodConfig import methodParams, methodFunctions
from util.util import objToDict

from ema_workbench import (ema_logging)

activateLogging = True
rootFolder = 'data'

runModel = {
    'dps': True,
    'plannedadaptive': True,
    'intertemporal': True
}
runMethod = {
    'mordm': True,
    'multi': True,
    'moro': True
}

if __name__ == '__main__':
    if (activateLogging):
        ema_logging.log_to_stderr(ema_logging.INFO)

    for key, model in models.items():
        if not runModel[key]:
            continue
        print('\nSTARTING: ' + model.name)

        for method, funcs in methodFunctions.items():
            if not runMethod[method]:
                continue
            print('----------------------------------')
            print(' ' + method.upper())
            print('----------------------------------')

            retVal = None
            for step in funcs['steps']:
                print(step)
                if retVal is None:
                    retVal = funcs[step](model, methodParams[method])
                else:
                    retVal = funcs[step](model, methodParams[method], retVal)
