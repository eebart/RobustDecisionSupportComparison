# Reference scenario selection for multi-scenario MORDM

There are three stages of analysis to select scenarios for multi-scenario MORDM.

## Stage 1: Build experiments to select from_dict

This work is found in `ScenarioSelection.ipynb`. This notebook contains the work to identify an initial set of experiments with which to select reference scenarios from in three ways: using the mean, median, and results from Prim analysis.

Instructions can be found in the notebook. Once this work is completed, move on to the second stage.

## Stage 2:

Due to large computational constraints, the maximally diverse set of alternatives is found using a Python script outside the Jupyter notebook environment.

To run this script and find the set of maximally diverse scenarios, follow these
instructions. The script requires four command line parameters, passed in
the order described.

1. model: The model name (one of 'intertemporal', 'plannedadaptive', and 'dps' in this case).
2. selection type: what selection method was used ('prim' in this case).
3. number of policies: the number of policies used to generate experiment data (10 in this case).
4. number of experiments: the number of experiments used to generate experiment data (500 in this case).
5. set size: the number of reference scenarios to identifiy (4 in this case).

These four parameters lead to the following command, for example:

```shell
python MaxDiverseSelect.py dps prim 10 500
```

The results of this script will be written into a file, which identifies which experiments in the larger set are considered maximally diverse. This information is used in the final step, where additional processing and visualization are completed.

## Stage 3: Final Processing
Includes additional visualizations and identification of the four selected reference scenarios that, along with the base reference scenario, are used to complete MORDM analysis. This can be found in `ScenarioVisualizatoin.ipynb`. 
