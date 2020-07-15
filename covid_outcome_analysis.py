import csv

FOLDER_PATH = './Covid_Data/'
OUTCOME_ICU_PATH = FOLDER_PATH + 'Outcome_ICU.csv'
COVID_UNSTRUCTURED_NOTES_PATH = FOLDER_PATH + 'Covid_Unstructured_Notes.csv'

admission_statistics = {}
with open(OUTCOME_ICU_PATH) as outcome_csv:
    csv_reader = csv.DictReader(outcome_csv)
    for row in csv_reader:
        adm_status = row['Admission_status']
        if adm_status in admission_statistics:
            admission_statistics[adm_status] += 1
        else:
            admission_statistics[adm_status] = 1
    print(admission_statistics)

total = 0
for key in admission_statistics:
    total += admission_statistics[key]

for key in admission_statistics:
    print(key, float(admission_statistics[key])/total)