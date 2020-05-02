import pandas as pd
import numpy as np

def calculate_euclidean_distance(point_1, point_2):
    """
    Given two points (np.array), calculate Euclidean distance
    """
    delta = point_1 - point_2
    return sum(delta**2)**0.5

def input_reader(filename):
	"""
	Read input csv file and return the input dataset as a numpy array
	"""
	dataset = [[1, 2, 3], [10, 30, 50], [100, 600, 700]]
	df = pd.read_csv(filename, header=None)
	return df.to_numpy()
