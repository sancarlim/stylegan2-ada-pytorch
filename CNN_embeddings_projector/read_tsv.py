#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : read_tsv.py
# Modified   : 01.02.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>

"""
Compute cosine distance between embeddings from tsv file.
"""

import pandas as pd
import csv
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt

metadata = csv.reader(open("/workspace/stylegan2-ada-pytorch/CNN_embeddings_projector/projections_vs_reals/00000/default/metadata.tsv"), delimiter="\t")
embeddings = list(csv.reader(open("/workspace/stylegan2-ada-pytorch/CNN_embeddings_projector/projections_vs_reals_nosprite/00000/default/tensors.tsv"), delimiter="\t"))

#embeddings already ordered from x1, to x1, from x2, to x2 ....
distances = []
for i in range(0,len(embeddings),2):
    emb_from = list(map(float, embeddings[i]))
    emb_to = list(map(float, embeddings[i+1]))
    distances.append( distance.cosine(emb_from,emb_to) )

textfile = open("/workspace/stylegan2-ada-pytorch/CNN_embeddings_projector/projections_vs_reals_nosprite/distances.txt", "w")
for element in distances:
    textfile.write(str(element) + "\n")
textfile.close()

distances = np.array(distances)
Q1 = np.quantile(distances, 0.25)
Q2 = np.quantile(distances, 0.5)
Q3 = np.quantile(distances, 0.75)
his = plt.hist(distances)
distances_indeces_ordered = np.argsort(distances) 
