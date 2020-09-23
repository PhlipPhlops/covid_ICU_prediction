# For the purposes of this, biased rows are any row in Covid_Unstructured_Notes.csv with a Result Verification Day
# within 7 days of its corresponding 'Time' recorded in the Outcome_ICU.csv
import pandas as pd

print("Running...")
FOLDER_PATH = '../Covid_Data/'
COVID_NOTES_OUTCOME_PATH = FOLDER_PATH + 'Same_Day_Concatenated_Notes.pkl'
BIAS_DELTA = '3 days'

notes = pd.read_pickle(COVID_NOTES_OUTCOME_PATH)

indices_to_drop = []

for index, row in notes.iterrows():
    notes_timestamp = row['Result Verification Day']
    outcome_timestamp = row['Admission Time']
    patient = row['Patient ID']
    
    # Self Iso patients have no Admission Status time
    # so set it to be the timestamp of the last day of their notes
    if row['Admission Status'] == 'Self Isolation':
        notes_timestamp = notes[notes["Patient ID"] == patient].iloc[-1]["Result Verification Day"]

    if (notes_timestamp - outcome_timestamp) < pd.Timedelta(BIAS_DELTA):
        indices_to_drop.append(index)


notes = notes.drop(indices_to_drop).reset_index()
print("Dropped {} notes".format(len(indices_to_drop)))

notes.to_pickle(FOLDER_PATH + 'SDC_Notes_minus_3day_bias.pkl')
