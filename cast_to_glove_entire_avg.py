import pandas as pd
import numpy as np
import load_glove as lg

vector_map = lg.vector_map
GLOVEd = 300
print("Assuming GloVe dimensionality is " + str(GLOVEd))

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
        'GloVe Avg': [],
        'Admission Status': []
    }
glove_df = pd.DataFrame(data=glove_notes)

for index, row in notes.iterrows():
    note_as_glove = []

    note = pad_punctuation(row['Concatenated Notes'])
    for word in note.split():
        # If word isn't represented by glove, use 0 vector of same size
        gloved_word = vector_map.get(word.lower(), [0.0]*GLOVEd)
        note_as_glove.append(gloved_word)

    updated_row = {
            'Patient ID': row['Patient ID'],
            'GloVe Avg': np.average(note_as_glove, axis=0),
            'Admission Status': row['Admission Status']
        }
    glove_df = glove_df.append(updated_row, ignore_index=True)

print('Saving GloVe average')
glove_df.to_csv('./Covid_Data/Concatted_Notes_GloVe_Averages_300d.csv')

#print("Saving GloVed notes to file")
#glove_df[['Patient ID', 'Notes in GloVe', 'Admission Status']].to_csv('./Covid_Data/Timeseries_Glove_Notes.csv')

# Refraining from padding and opting for a batch size of one
#print('Padding GloVe\'d notes to consistent length')
#padded_notes = {
#        'Patient ID': [],
#        'Padded GloVe Notes': [],
#        'Admission Status': []
##    }
#padded_df = pd.DataFrame(data=padded_notes)
## Find max length of note
#max_length = 0
#for index, row in glove_df.iterrows():
#    curr_length = len(row['Notes in GloVe'])
##    if curr_length > max_length: max_length = curr_length
##
#print("Max length of notes is {}".format(max_length))
## Pad row with empty glove vectors to be same size as the max number of words in a note
#for index, row in glove_df.iterrows():
###    notes = row['Notes in GloVe']
#    notes.extend([[0.0]*GLOVEd] * (max_length - len(notes)))
#    
#    updated_row = {
#            'Patient ID': row['Patient ID'],
##            'Padded GloVe Notes': notes,
#            'Admission Status': row['Admission Status']
#        }
#    padded_df = padded_df.append(updated_row, ignore_index=True)
#
#print('Saving padded GloVe notes to file')
#padded_df.to_csv('./Covid_Data/Time_Series_GloVe_Notes_2.csv')
