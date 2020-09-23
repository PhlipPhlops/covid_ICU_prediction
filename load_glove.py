import sys
import numpy as np
from scipy import spatial

# script from: https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
vector_map = {}

GLOVEd = 50

with open("./GloVe/glove.6B.{}d.txt".format(GLOVEd), 'r', encoding='utf-8') as f:
    print("loading word vectors {}d...".format(GLOVEd))
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        vector_map[word] = vector
    # Done, print newline
    print("...done")

def find_closest_embeddings(embedding):
    return sorted(vector_map.keys(), key=lambda word: spatial.distance.euclidean(vector_map[word], embedding))
