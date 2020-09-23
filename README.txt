Given the original dataseti (Unstructured notes including PID and a separate Outcome file also including PID), the following is the order of the scripts to be run to turn the original notes data into unbiased GloVe vectors for training
-- Note; the filenames within the scripts are not automatically configured. Verify the filenames pointed to in each script before running to making sure you're reading from- and writing to- the file you intend to.

1. remove_unused_columns.py
2. merge_notes_and_outcomes.py
3. drop_biased_rows.py
4. preprocess_notes.py
5. cast_notes_to_glove.py
