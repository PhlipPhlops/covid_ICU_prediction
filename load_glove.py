import sys
import numpy as np
from scipy import spatial

# script from: https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
vector_map = {}

with open("./GloVe/glove.6B.50d.txt", 'r') as f:
    print("loading word vectors 50d...")
    num_lines = 400000
    curr_line = 1 
    for line in f:
        # Print Status
        status = "\r" + str(curr_line) + "/" + str(num_lines) + "            "
        curr_line += 1
        sys.stdout.write(status)
        # Actual Embedding
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        vector_map[word] = vector
    # Done, print newline
    sys.stdout.write('\n')

def find_closest_embeddings(embedding):
    return sorted(vector_map.keys(), key=lambda word: spatial.distance.euclidean(vector_map[word], embedding))
