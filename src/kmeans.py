import random
import numpy as np
    

def find_closest_centroid(point, centroids):
    min_dist = np.inf
    min_centroid_inx = -1
    for centroid_inx in range(len(centroids)):
        dist = calculate_euclidean_distance(point, centroids[centroid_inx])
        # print("point: {}, centroid: {}, dist: {}".format(point, centroids[centroid_inx], dist))
        if dist < min_dist:
            min_centroid_inx = centroid_inx
            min_dist = dist
    return min_centroid_inx


def k_means(dataset, k, centroids):
    stop_threshold = 0.01
    if len(centroids) == 0:
        # initilization step
        centroids = random.sample(dataset, k)
        # print("initial centroids: {}".format(centroids))
    cluster_assignment = {}
    # cluster assignment
    for point in dataset:
        # print("before calling find best centroid centroids: {}".format(centroids))
        cluster_inx = find_closest_centroid(point, centroids)
        # print("point: {}, cluster_inx: {}".format(point, cluster_inx))
        try:
            cluster_assignment[cluster_inx].append(point)
        except:
            cluster_assignment[cluster_inx] = [point]
    # print("cluster assignment: {}".format(cluster_assignment))
    # update my centroids
    new_centroids = []
    for centroid_inx, points in cluster_assignment.items():
        if points:
            new_centroid = np.mean(np.array(points),0)
        # print("points: {} new centroid: {}".format(points, new_centroid))
        new_centroids.append(new_centroid)
    print("centroids: {}".format(centroids))
    print("new_centroids: {}".format(new_centroids))
    # test for stopping the iteration
    stop = True
    for inx in range(len(centroids)):
        dist = calculate_euclidean_distance(centroids[inx], new_centroids[inx])
        print("dist: {}".format(dist))
        if dist > stop_threshold:
            stop = False
            break
        print("stop: {}".format(stop))
    if stop:
        print("stop condition --> cluster_assignment: {}".format(cluster_assignment))
        # centroid have not been changed so much
        return cluster_assignment
    return k_means(dataset, k, new_centroids)
    
    
        

dataset = [[1, 2, 3], [10, 30, 50], [100, 600, 700]]
cluster_assignment_ = k_means(dataset, 2, [])
print("cluster assignment: {}".format(cluster_assignment_))
# dist = calculate_euclidean_distance([1,2,3], [1,2,5])
# print(dist)