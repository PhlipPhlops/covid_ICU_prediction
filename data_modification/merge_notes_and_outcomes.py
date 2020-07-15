import pandas as pd

FOLDER_PATH = '../Covid_Data/'
COVID_NOTES_PATH = FOLDER_PATH + 'Covid_Notes_w_PID.csv'
OUTCOMES_PATH = FOLDER_PATH + 'Outcome_ICU.csv'

notes = pd.read_csv(COVID_NOTES_PATH)
outcomes = pd.read_csv(OUTCOMES_PATH)

# Match column types for the merge
outcomes['Patient'] = outcomes['Patient'].apply(str)

merged = pd.merge(notes, outcomes, left_on="Patient ID", right_on="Patient")

# Only store relevant columns
merged_clean = merged[["Patient ID", "Document Name", "Result Verification Day", "Note Result", "Admission_status", "Time"]]
merged_clean = merged_clean.rename(columns={'Time': 'Admission_time'})

merged_clean.to_csv('../Covid_Data/Covid_Notes_with_Outcomes_2.csv')
