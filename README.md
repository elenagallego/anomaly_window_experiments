# anomaly_window_experiments
Code for context window experiments on time series anomaly detection.

# Anomaly Detection Window Experiments

This repository contains code developed for a Master's thesis exploring the role of different characteristics for context window in time series anomaly detection. The experiments use datasets from the UCR Anomaly Archive and evaluate detection performance under different window configurations.

## Contents

| File                          | Description                                                                                                       |
|------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `size_experiment.py`         | Evaluates detection performance across normalized window sizes.                                                    |
| `weight_experiment1.py`      | Determine the optimal context window size for each dataset group to enable the weighting analysis.                 |
| `weight_experiment2.py`      | Evaluates detection performance across different weight slopes.                                                    |
| `position_experiment.py`     | Evaluates detection performance across different context positions in the training data of cyclical datasets.      |
| `prototype_experiment.py`    | Evaluates detection performance using a prototype-based window selection approach.                                 |

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt


### Datasets
This repository uses time series datasets from the [UCR Anomaly Archive](https://www.cs.ucr.edu/~eamonn/discords/).

Due to size and licensing constraints, **the datasets are not included in this repository**.  
To run the experiments:
Download the required `.txt` files manually from the [UCR Anomaly Archive](https://www.cs.ucr.edu/~eamonn/discords/).

