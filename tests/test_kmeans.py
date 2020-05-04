import numpy as np
import pytest

from src.models.kmeans import KMeans


@pytest.fixture
def kmeans_mock():
    return KMeans(None,3,0.01) 

@pytest.fixture
def kmeans_mock_with_data():
    dataset = np.array([[1, 2, 3], [10, 30, 50], [100, 600, 700]])
    return KMeans(dataset,3)

def test_find_closest_centroid(kmeans_mock):
    centorids = [np.array([10,20,30]), np.array([-10,-20,-30]), np.array([1,2,3])]
    point = np.array([1.1,2.1,3.1])
    centroid_inx = kmeans_mock.find_closest_centroid(point, centorids)
    assert centroid_inx == 2

def test_find_closest_centroid_with_two_centroid_candidates(kmeans_mock):
    centorids = [np.array([3,3,3]), np.array([1,1,1]), np.array([10,11,12])]
    point = np.array([2,2,2])
    centroid_inx = kmeans_mock.find_closest_centroid(point, centorids)
    assert centroid_inx == 0

def test_initialize_centroids(kmeans_mock_with_data):
    centroids = kmeans_mock_with_data.initialize_centroids()
    assert len(centroids) == 3

def test_update_centroids(kmeans_mock):
    clusters_assignment = {0: [np.array([1,2,3]), np.array([10,20,30])], 
                            1: [np.array([-11,20,300])], 2: [np.array([0,1,13]), np.array([2,2.1,3.2])]}
    centroids = kmeans_mock.update_centroids(clusters_assignment)
    assert len(centroids) == 3
    assert np.array_equal(centroids[0], [5.5, 11.0 , 16.5])
    assert np.array_equal(centroids[1], [-11, 20, 300])
    assert np.array_equal(centroids[2], [1.0, 1.55, 8.1])

def test_is_close_enough_true(kmeans_mock):
    old_centroids = np.array([[2, 4, 6], [6, 8, 10]])
    new_centroids = np.array([[2.01, 4.0, 6.0], [6, 8, 10]])
    assert kmeans_mock.is_close_enough(old_centroids, new_centroids)

def test_is_close_enough_false(kmeans_mock):
    old_centroids = np.array([[2, 4, 6], [6, 8, 10]])
    new_centroids = np.array([[2.01, 4.000001, 6.0], [6, 8, 10]])
    assert not kmeans_mock.is_close_enough(old_centroids, new_centroids)
