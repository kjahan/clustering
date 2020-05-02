
def calculate_euclidean_distance(point_1, point_2):
    """
    Given two points (np.array), calculate Euclidean distance
    """
    delta = point_1 - point_2
    return sum(delta**2)**0.5