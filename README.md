# clustering
This project is a collection of clustering algorithms

K-Means
=======

## General description
 
This project is a Python implementation of k-means clustering algorithm for vectorized input

## Requirements

You should setup the conda environment (i.e. kmeans) using the environment.yml file:

`conda env create -f environment.yml`

## Activate conda environment:

`conda activate kmeans`

(Run `unset PYTHONPATH` on Mac OS)


## Input

A list of points in an n-dimensional space.

## Output

The clusters of points.  By default we stores the computed clusters into a csv file: `output.csv`.  You can specify your output filename using `--output` argument option.

## How to run:

`python -m src.run --input YOUR_DATA --clusters CLUSTERS_NO`

Note that the runner expects the input dataset file to be in `data` folder.


## Run tests

`python -m pytest tests/`

# To deactivate the conda environment:

`conda deactivate`
