import numpy as np
import pytest

import src.utilities as utilities


@pytest.fixture
def distinct_points_mock():
	point_1 = np.array([1, 2, 3])
	point_2 = np.array([10, 20, 30])
	return (point_1, point_2)

@pytest.fixture
def same_points_mock():
	point_1 = np.array([1.1, 20.4, 30.2, 300])
	return (point_1, point_1)

def test_calculate_euclidean_distance_for_distinct_points(distinct_points_mock):
	distance = utilities.calculate_euclidean_distance(distinct_points_mock[0], distinct_points_mock[1])
	assert distance == pytest.approx(33.67, 0.01)

def test_calculate_euclidean_distance_for_same_points(same_points_mock):
	distance = utilities.calculate_euclidean_distance(same_points_mock[0], same_points_mock[1])
	assert distance == 0