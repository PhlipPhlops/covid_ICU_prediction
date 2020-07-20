# First we want to get a better sense of where the glove vectors are failing to cast
# and what we can do about it. So let's print uncastable words and their context
import sys
import pandas as pd
import numpy as np
import load_glove as lg
import random
import pprint
pp = pprint.PrettyPrinter(indent=4)

vector_map = lg.vector_map

notes = pd.read_csv('./Covid_Data/CNwO_minus_3day_bias.csv')
notes['Note Result'] = notes['Note Result'].astype(str)

# List of punctuation
PUNCTS = "!\"#$%'()*+,-./:;<=>?@[\]^_`{|}~"
def pad_punctuation(note):
    # Places spaces around puncatuations so they'll be
    # separated by a .split() method and recognized
    # by GloVe
    updated_list = []
    for i in range(len(note)):
        if note[i] in PUNCTS:
            updated_list.extend([" ", note[i], " "])
        else:
            updated_list.append(note[i])
    return ''.join(updated_list)

# Num words to display before and after
CONTEXT_WINDOW = 3
failure_dict = {}
for index, row in notes.iterrows():
    # Quickest way to remove punctuation
    # Also removes Dates
    # note_sans_punct = str.translate(row['Note Result'], transtable)
    note = pad_punctuation(row['Note Result'])

    note_split = note.split()
    failure_dict[index] = {}
    for i in range(len(note_split)):
        word = note_split[i]
        # If word isn't represented by glove, use 0 vector of same size
        if not word.lower() in vector_map:
            # Calculate word window
            words_back = i - CONTEXT_WINDOW
            words_forward = i + CONTEXT_WINDOW
            if words_back < 0: words_back = 0
            if words_forward >= len(note_split): words_forward = len(note_split)

            failure_dict[index][i] = [note_split[i], " ".join(note_split[words_back:words_forward])]
    if failure_dict[index] == {}: failure_dict.pop(index, None)


# Print a randsom sample of errors
subset_dict = {}
random.shuffle(failure_dict.keys())
for key in failure_dict.keys()[:100]:
    subset_dict[key] = failure_dict[key]
pp.pprint(subset_dict)
