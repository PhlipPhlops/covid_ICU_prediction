import pandas as pd
import numpy as np
import load_biowordvec as lb

model = lb.model
BIOVECd = 200
print("Assuming BioWordVec dimensionality is " + str(BIOVECd))

notes = pd.read_csv('./Covid_Data/Concatenated_Notes.csv')

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

glove_notes = {
        'Patient ID': [],
        'BioWordVec Avg': [],
        'Admission Status': []
    }
glove_df = pd.DataFrame(data=glove_notes)

for index, row in notes.iterrows():
    note_as_biowordvec = []

    note = pad_punctuation(row['Concatenated Notes'])
    for word in note.split():
        # If word isn't represented by glove, use 0 vector of same size
        try:
            biovecced_word = model.get_vector(word.lower())
        except KeyError:
            biovecced_word = np.array([0.0]*BIOVECd)
        note_as_biowordvec.append(biovecced_word)

    updated_row = {
            'Patient ID': row['Patient ID'],
            'BioWordVec Avg': np.average(note_as_biowordvec, axis=0),
            'Admission Status': row['Admission Status']
        }
    glove_df = glove_df.append(updated_row, ignore_index=True)

print('Saving BioWordVec average')
glove_df.to_csv('./Covid_Data/Concatted_Notes_BioWordVec_averages_200d.csv')

