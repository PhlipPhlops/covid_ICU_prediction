import pandas as pd

notes = pd.read_csv('../Covid_Data/out_of_focus/Covid_Notes_with_Outcomes.csv')
notes["Result Verification Day"] = pd.to_datetime(notes["Result Verification Day"])
notes["Admission_time"] = pd.to_datetime(notes["Admission_time"])
patients = notes['Patient ID'].unique()

# Ensure type is string
notes['Note Result'] = notes['Note Result'].astype(str)

pre_df = {
        "Patient ID": [],
        "Daily Notes": [],
        "Result Verification Day": [],
        "Admission Status": [],
        "Admission Time": [],
    }
condensed_df = pd.DataFrame(data=pre_df)

for patient in patients:
    pat_notes = notes.loc[notes['Patient ID'] == patient]
    
    # Sort notes in ascending order (oldest to newest)
    pat_notes = pat_notes.sort_values(by="Result Verification Day")

    # Take a row to reference unchanging fields
    patient_data = pat_notes.iloc[0]

    unique_days = pat_notes['Result Verification Day'].unique()

    for day in unique_days:
        note_store = []
        daily_rows = pat_notes[pat_notes["Result Verification Day"] == day]
        
        # Take a row for referencing unchanging fields
        daily_data = daily_rows.iloc[0]
        
        for index, row in daily_rows.iterrows():
            note_store.extend([
                row['Result Verification Day'].strftime('%m-%d-%y'),
                row['Document Name'],
                row['Note Result']
            ])

        concatted_note = ' '.join(note_store)
        updated_daily_row = {
                "Patient ID": patient_data["Patient ID"],
                "Daily Notes": concatted_note,
                "Result Verification Day": daily_data["Result Verification Day"],
                "Admission Status": patient_data["Admission_status"],
                "Admission Time": daily_data["Admission_time"]
            }
        condensed_df = condensed_df.append(updated_daily_row, ignore_index=True)

condensed_df.to_pickle('../Covid_Data/Same_Day_Concatenated_Notes.pkl')
