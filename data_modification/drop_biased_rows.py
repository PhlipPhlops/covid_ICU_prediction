# For the purposes of this, biased rows are any row in Covid_Unstructured_Notes.csv with a Result Verification Day
# within 7 days of its corresponding 'Time' recorded in the Outcome_ICU.csv
import pandas as pd
from datetime import datetime

print("Running...")
FOLDER_PATH = '../Covid_Data/'
COVID_NOTES_OUTCOME_PATH = FOLDER_PATH + 'Covid_Notes_with_Outcomes.csv'
DAYS_WITHIN_BIAS = 3

notes = pd.read_csv(COVID_NOTES_OUTCOME_PATH)
# Force rows into strings for consistency (some are floats)
notes['Result Verification Day'] = notes['Result Verification Day'].astype(str)
notes['Admission_time'] = notes['Admission_time'].astype(str)

indices_to_drop = []
num_null_indices = 0
admission_stats = {
        'ICU': 0,
        'Hospitalized': 0,
        'Self Isolation': 0,
    }
nan_count = {
        'note': 0,
        'outcome': 0
    }
for index, row in notes.iterrows():
    notes_date_string = row['Result Verification Day']
    outcome_date_string = row['Admission_time']
    if notes_date_string == 'nan' or outcome_date_string == 'nan':
        if notes_date_string == 'nan':
            nan_count['note'] += 1
        else:
            nan_count['outcome'] += 1
        ### All of these no-date notes seem to be Self Isolation (50,176)
        ### so I'm refraining from dropping them
        # admission_stats[row['Admission_status']] += 1
        # indices_to_drop.append(index)
        # num_null_indices += 1
        continue

    notes_datetime = datetime.strptime(notes_date_string, '%m/%d/%Y')
    outcome_datetime = datetime.strptime(outcome_date_string, '%m/%d/%y')
    if (notes_datetime - outcome_datetime).days < DAYS_WITHIN_BIAS:
        admission_stats[row['Admission_status']] += 1
        indices_to_drop.append(index)

print("Notes before drop: {}".format(len(notes)))
notes = notes.drop(indices_to_drop).reset_index()
print("Notes after drop: {}".format(len(notes)))
print("Dropped {} notes".format(len(indices_to_drop)))
print("Number of rows with null date columns: {}".format(num_null_indices))

print("Status stats of dropped patients: {}".format(admission_stats))
print("Nan Count: {}".format(nan_count))

notes.to_csv(FOLDER_PATH + 'CNwO_minus_3day_bias.csv', index=False)
