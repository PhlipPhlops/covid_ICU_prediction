# For the purposes of this, biased rows are any row in Covid_Unstructured_Notes.csv with a Result Verification Day
# within 7 days of its corresponding 'Time' recorded in the Outcome_ICU.csv
import pandas as pd

FOLDER_PATH = '../Covid_Data/'
COVID_UNSTRUCTURED_NOTES_THINNED_PATH = FOLDER_PATH + 'Covid_Unstructured_Notes_Thinned.csv'
OUTCOMES_ICU_PATH = FOLDER_PATH + 'Outcome_ICU.csv'
MAPPING_PATH = FOLDER_PATH + 'All_Covid_Pos_EMPI.csv'

notes = pd.read_csv(COVID_UNSTRUCTURED_NOTES_THINNED_PATH)
outcomes = pd.read_csv(OUTCOMES_ICU_PATH)
mapping = pd.read_csv(MAPPING_PATH)

print(outcomes.head())
print(notes.head())
print(mapping.head())

for m_index, m_row in mapping.iterrows():
    patient_id = m_row['Patient ID']
    patient_empi = int(m_row['Patient EMPI Nbr'])
    corr_notes = notes.loc[notes['Patient EMPI Nbr'] == "2673698"]
    corr_outcomes = outcomes.loc[outcomes['Patient'] == patient_id]
    print("------- ", m_index, " -------")
    print(corr_notes)
    print(corr_outcomes)
