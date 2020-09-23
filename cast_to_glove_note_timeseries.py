import pandas as pd
import numpy as np
import load_glove as lg

vector_map = lg.vector_map
GLOVEd = 50
print("Assuming GloVe dimensionality is " + str(GLOVEd))

notes = pd.read_pickle('./Covid_Data/SDC_Notes_minus_3day_bias.pkl')

# List of punctuation
def pad_punctuation(note):
    # Places spaces around puncatuations so they'll be
    # separated by a .split() method and recognized
    # by GloVe
    PUNCTS = "!\"#$%'()*+,-./:;<=>?@[\]^_`{|}~"
    updated_list = []
    for i in range(len(note)):
        if note[i] in PUNCTS:
            updated_list.extend([" ", note[i], " "])
        else:
            updated_list.append(note[i])
    return ''.join(updated_list)

# Cast Notes to GloVe Averages
glove_notes = {
        'Patient ID': [],
        'GloVe Avg': [],
        'Result Verication Day': [],
        'Admission Status': []
    }
glove_df = pd.DataFrame(data=glove_notes)
for index, row in notes.iterrows():
    if type(row['Daily Notes']) != str: continue
    
    note_as_glove = []
    note = pad_punctuation(row['Daily Notes'])
    for word in note.split():
        # If word isn't represented by glove, use 0 vector of same size
        gloved_word = vector_map.get(word.lower(), [0.0]*GLOVEd)
        note_as_glove.append(gloved_word)

    updated_row = {
            'Patient ID': row['Patient ID'],
            'GloVe Avg': np.average(note_as_glove, axis=0),
            'Result Verification Day': row['Result Verification Day'],
            'Admission Status': row['Admission Status']
        }
    glove_df = glove_df.append(updated_row, ignore_index=True)


'''Segment timeseries into 4 averages
    From recent to oldest
    [
        0: Average of the remaining notes older than a month
        1: Average of the previous month
        2: Average of the previous week
        3: Average of the most recent available day
    ]
'''

print("Joining GloVed notes into timeseries")
patient_timeseries_df = {
        'Patient ID': [],
        'Day Avg Timeseries': [],
        'Admission Status': []
    }
patient_timeseries_df = pd.DataFrame(data=patient_timeseries_df)
patients = glove_df['Patient ID'].unique()
for patient in patients:
    patient_df = glove_df[glove_df['Patient ID'] == patient]
    pat_data = patient_df.iloc()[0]
    
    # Contains all notes
    note_buckets = [[], [], [], []]
    bucket_thresholds = [365, 30, 7, 0]

    latest_timestamp = patient_df.iloc[-1]['Result Verification Day']
    for index, row in patient_df.iterrows():
        current_timestamp = row['Result Verification Day']
        days_dif = (latest_timestamp - current_timestamp).days

        # Set pointer to array to append
        # Defaults to an unrecorded array
        array_to_append = []
        for i in range(len(bucket_thresholds)):
            if days_dif <= bucket_thresholds[i]:
                array_to_append = note_buckets[i]
        array_to_append.append(row['GloVe Avg'])

        ''' Aux code'''
        ### Add empty vectors to represent days between timestamps
        #if days_dif > 1:
        #    spacing = [[0.0]*GLOVEd] * (days_dif - 1)
        #    note_buckets.extend(spacing)

    note_timeseries = []
    for bucket in note_buckets:
        bucket_average = np.average(np.array(bucket), axis=0).tolist()
        note_timeseries.append(bucket_average)

    updated_row = {
            'Patient ID': pat_data['Patient ID'],
            'Bucket Timeseries': note_timeseries,
            'Admission Status': pat_data['Admission Status']
        }
    patient_timeseries_df = patient_timeseries_df.append(updated_row, ignore_index=True)


print("Saving GloVed Note Timeseries to file")
patient_timeseries_df.to_pickle('./Covid_Data/Bucket_Timeseries.pkl')
print("Done!")

### Auxiliary Code ###

#print("Padding timeseries to be of equal length")
## Find max length of note
#max_length = 0
#for index, row in patient_timeseries_df.iterrows():
#    curr_length = len(row['Day Avg Timeseries'])
#    if curr_length > max_length: max_length = curr_length
#
#print("Max length of notes is {}".format(max_length))
## Pad row with empty glove vectors to be same size as the max number of words in a note
#output_dict = {
#        'Patient ID': [],
#        'Day Avg Timeseries': [],
#        'Admission Status': []
#    }
#output_df = pd.DataFrame(data=output_dict)
#for index, row in patient_timeseries_df.iterrows():
#    # PREPREND padding to timeseries
#    notes = row['Day Avg Timeseries']
#    padding = [[0.0]*GLOVEd] * (max_length - len(notes))
#    padding.extend(notes)
#    notes = padding
#    # Replace timeseries in df with extended note
#    updated_row = {
#            'Patient ID': row['Patient ID'],
#            'Day Avg Timeseries': np.array(notes),
#            'Admission Status': row['Admission Status']
#        }
#    output_df = output_df.append(updated_row, ignore_index=True)


