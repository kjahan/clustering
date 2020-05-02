import numpy as np
import logging

from src.utilities import calculate_euclidean_distance

logging.basicConfig(filename='kmeans.log',level=logging.DEBUG)

class KMeans:
    def __init__(self, dataset, clusters_no, stop_threshold=0.01):
        self.dataset = dataset
        self.k = clusters_no
        self.stop_threshold = stop_threshold
        self.clusters = {}

    def find_closest_centroid(self, point, centroids):
        min_dist = np.inf
        min_centroid_inx = -1
        for centroid_inx in range(len(centroids)):
            dist = calculate_euclidean_distance(point, centroids[centroid_inx])
            logging.info('point: {}, centroid: {}, dist: {}'.format(point, centroids[centroid_inx], dist))

            if dist < min_dist:
                min_centroid_inx = centroid_inx
                min_dist = dist
        return min_centroid_inx


    def k_means(self, centroids):
        if len(centroids) == 0:
            # initilization step
            idx = np.random.randint(self.dataset.shape[0], size=self.k)
            centroids = self.dataset[idx,:]
            logging.info('initial centroids: {}'.format(centroids))

        clusters_assignment = {}
        # cluster assignment
        for point in self.dataset:
            cluster_inx = self.find_closest_centroid(point, centroids)
            logging.info('point: {}, cluster_inx: {}'.format(point, cluster_inx))
            try:
                clusters_assignment[cluster_inx].append(point)
            except:
                clusters_assignment[cluster_inx] = [point]
        logging.info('cluster assignment: {}'.format(clusters_assignment))
        # update centroids
        new_centroids = []
        for centroid_inx, points in clusters_assignment.items():
            if points:
                new_centroid = np.mean(np.array(points),0)
            logging.info('points: {} new centroid: {}'.format(points, new_centroid))
            new_centroids.append(new_centroid)
        logging.info('centroid: {}'.format(centroids))
        logging.info('new centroid: {}'.format(new_centroids))
        # test for stopping the iteration
        stop = True
        for inx in range(len(centroids)):
            dist = calculate_euclidean_distance(centroids[inx], new_centroids[inx])
            logging.info('dist: {}'.format(dist))
            if dist > self.stop_threshold:
                stop = False
                break
            logging.info('stop: {}'.format(stop))
        if stop:
            logging.info('stop condition --> clusters_assignment: {}'.format(clusters_assignment))
            # centroid have not been changed so much
            return clusters_assignment
        return self.k_means(new_centroids)

    def fit(self):
        self.clusters = self.k_means([])
