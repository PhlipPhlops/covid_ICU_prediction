# A set of convenience methods to load in patient CSVs
# Provides access to vairables containing patient data
import pandas as pd

FILE_PATH = './Covid_Data/'

def load_mapping_frame():
    # Imports the csv that maps patients outcome to note EMPI
    return pd.read_csv(FILE_PATH + 'All_Covid_Pos_EMPI.csv')

def load_notes(note_state='thinned'):
    file_name = '';
    if note_state == 'no_mod':
        file_name = 'Covid_Unstructured_Notes.csv'
    elif note_state == 'thinned':
        file_name = 'Covid_Unstructured_Notes_Thinned.csv'
    else:
        # Update this list and options after adding new note modificaitons
        print('No such note state. Options are: {no_mod, thinned}')
        return
    print('Loading ' + FILE_PATH + file_name)
    return pd.read_csv(FILE_PATH + file_name)

def load_notes_sample():
    return pd.read_csv(FILE_PATH + 'Notes_Sample.csv')

def load_outcome():
    return pd.read_csv(FILE_PATH + 'Outcome_ICU.csv')
