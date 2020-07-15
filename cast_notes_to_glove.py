import pandas as pd
import load_glove as lg
import numpy as np
from string import maketrans

vector_map = lg.vector_map
GLOVEd = 50
print("Assuming Glove d is " + str(GLOVEd))

notes = pd.read_csv('./Covid_Data/Covid_Notes_with_Outcomes.csv')

glove_notes = {'Note in GloVe': [], 'GloVe Avg': []}

# List of punctuation to clean from notes
punctuation = "!\"#$%'()*+,-./:;<=>?@[\]^_`{|}~"
empty_replacement = " " * len(punctuation)
transtable = maketrans(punctuation, empty_replacement)

for index, row in notes.iterrows():
    note_as_glove = []
    if type(row['Note Result']) != str:
        # Some notes aren't strings as expected
        print("Not a strin; index: {}, data: {}".format(index, row['Note Result']))
        row['Note Result'] = ""
    # Quickest way to remove punctuation
    note = str.translate(row['Note Result'], transtable)
    for word in note.split():
        # If word isn't represented by glove, use 0 vector of same size
        gloved_word = vector_map.get(word.lower(), [0.0]*GLOVEd)
        note_as_glove.append(gloved_word)
    glove_notes['Note in GloVe'].append(note_as_glove)
    glove_notes['GloVe Avg'].append(np.average(note_as_glove, axis=0))

# Save to file
glove_frame = pd.DataFrame(data=glove_notes)
merged = notes.merge(glove_frame, left_index=True, right_index=True)
merged.to_csv('./Covid_Data/Notes_w_GloVe_and_Outcome.csv')

