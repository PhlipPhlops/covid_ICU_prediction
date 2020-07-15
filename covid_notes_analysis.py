import csv

FOLDER_PATH = './Covid_Data/'
OUTCOME_ICU_PATH = FOLDER_PATH + 'Outcome_ICU.csv'
COVID_UNSTRUCTURED_NOTES_PATH = FOLDER_PATH + 'Covid_Unstructured_Notes.csv'

with open(COVID_UNSTRUCTURED_NOTES_PATH) as outcome_csv:
    csv_reader = csv.DictReader(outcome_csv)
    line_count = 0
    for row in csv_reader:
        if line_count % 100000 == 0:
            print(row)
        line_count += 1
    print("Notes count: ", line_count)


admission_statistics = {}
with open(OUTCOME_ICU_PATH) as outcome_csv:
    csv_reader = csv.DictReader(outcome_csv)
    line_count = 0
    for row in csv_reader:
       line_count += 1
    print("Outcome count: ", line_count)
