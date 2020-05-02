import os
import argparse

import src.utilities as utilities
from src.models.kmeans import KMeans

def run(input_fn, output_fn, k):
    dataset = utilities.input_reader(input_fn)
    model = KMeans(dataset, k)
    model.fit()
    print("Computed clusters: {}".format(model.clusters))
    utilities.save(model.clusters, output_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run clustering for vecorized data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='kmeans',
                        dest='model', help='clustering model to run')
    parser.add_argument('--input', type=str, default='input.csv', dest='input', 
                        help='input location file name')
    parser.add_argument('--output', type=str, default='output.csv', dest='output', 
                        help='clusters output file name')
    parser.add_argument('--clusters', type=int, default=3, dest='clusters', help='number of clusters')
    args = parser.parse_args()
    input_fn = os.path.join("data", args.input)
    output_fn = args.output
    run(input_fn, output_fn, args.clusters)
