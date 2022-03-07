# Last Modified   : 01.02.2022
# By              : Sandra Carrasco <sandra.carrasco@ai.se>

"""
Compute cosine distance between embeddings from tsv file.
"""

import csv
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--metadata", type=str,
                        help='path to metadata file')
    parser.add_argument("--embeddings_path", type=str, default=None,
                        help='path to embeddings saved as tensors.tsv')
    parser.add_argument("--save_path", type=str,
                        help='path to save distances in text file')
    args = parser.parse_args()

    metadata = csv.reader(open(args.metadata), delimiter="\t")
    embeddings = list(csv.reader(open(args.embeddings_path), delimiter="\t"))

    # embeddings already ordered from x1, to x1, from x2, to x2 ....
    distances = []
    for i in range(0, len(embeddings), 2):
        emb_from = list(map(float, embeddings[i]))
        emb_to = list(map(float, embeddings[i + 1]))
        distances.append(distance.cosine(emb_from, emb_to))

    textfile = open(args.save_path, "w")
    for element in distances:
        textfile.write(str(element) + "\n")
    textfile.close()

    distances = np.array(distances)
    Q1 = np.quantile(distances, 0.25)
    Q2 = np.quantile(distances, 0.5)
    Q3 = np.quantile(distances, 0.75)
    his = plt.hist(distances)
    distances_indeces_ordered = np.argsort(distances)
