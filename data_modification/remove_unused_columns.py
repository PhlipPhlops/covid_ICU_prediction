import pandas as pd

FOLDER_PATH = '../Covid_Data/'
COVID_UNSTRUCTURED_NOTES_PATH = FOLDER_PATH + 'Covid_Unstructured_Notes_PID.csv'

notes_raw = pd.read_csv(COVID_UNSTRUCTURED_NOTES_PATH)
# Only Necessary Columns
notes_cooked = notes_raw[["Patient ID", "Document Name", "Result Verification Day", "Note Result"]]
notes_cooked.to_csv(FOLDER_PATH + "Covid_Notes_w_PID.csv", index=False)
