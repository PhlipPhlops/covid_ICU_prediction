import pandas as pd

notes_orig = pd.read_csv('../Covid_Data/CNwO_minus_3day_bias.csv')
patients = notes_orig['Patient ID'].unique()

# Ensure type is string
notes_orig['Note Result'] = notes_orig['Note Result'].astype(str)

pre_df = {
        "Patient ID": [],
        "Concatenated Notes": [],
        "Admission Status": []
    }
condensed_df = pd.DataFrame(data=pre_df)

for patient in patients:
    pat_notes = notes_orig.loc[notes_orig['Patient ID'] == patient]
    # For copying over only one instance of redundant data
    patient_data = pat_notes.iloc[0]

    note_store = []
    for index, row in pat_notes.iterrows():
        note_store.extend([
            row['Result Verification Day'],
            row['Document Name'],
            row['Note Result']
        ])

    concatted_note = ' '.join(note_store)
    updated_patient_row = {
            "Patient ID": patient_data["Patient ID"],
            "Concatenated Notes": concatted_note,
            "Admission Status": patient_data["Admission_status"]
        }
    condensed_df = condensed_df.append(updated_patient_row, ignore_index=True)

condensed_df.to_csv('../Covid_Data/Concatenated_Notes.csv', index=True)
