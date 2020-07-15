import pandas as pd

FOLDER_PATH = '../Covid_Data/'
OUTCOMES_PATH = FOLDER_PATH + 'Outcome_ICU.csv'
MAPPING_PATH = FOLDER_PATH + 'All_Covid_Pos_EMPI.csv'

outcomes = pd.read_csv(OUTCOMES_PATH)
mapping = pd.read_csv(MAPPING_PATH)
# notes_cooked = notes_raw[["Patient EMPI Nbr", "Document Name", "Result Verification Day", "Note Result"]]
# notes_cooked.to_csv(FOLDER_PATH + "Covid_Unstructured_Notes_Thinned.csv", index=False)
joined = outcomes.merge(mapping, left_on='Patient', right_on='Patient ID')
print(joined.head())
