import pandas as pd
import numpy as np
import random
import csv


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

def save(clusters_assignment, filename="output.csv"):
    """
    Taking a cluster assignment Dict and storing in a CSV file.
    """
    if not clusters_assignment:
        return
    cluster_ids = list(clusters_assignment.keys())
    point_dimension = clusters_assignment[cluster_ids[0]][0].shape[0]
    header = ['dim_' + str(inx) for inx in range(point_dimension)]
    header += ['cluster_id']
    # save clusters into a csv file
    with open(filename, mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for cluster_id, points in clusters_assignment.items():
            for point in points:
                point_list = list(point) + [cluster_id]
                writer.writerow(point_list)
