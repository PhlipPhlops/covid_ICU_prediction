The filenames in the output folders (training_output, classification reports) describe the architecture of the network used to generate them. It's not entirely consistent.
For example, under classification_reports: the file 100d_thin_earlystop_200epoch50i50hn.csv describes a networking taking in 100d GlOVe vectors as an endpoint, trained over 200epoch (with early stopping) configured with 50i (50 input neurons) and 50hn (50 hidden neurons). Every additional word is just a tag to describe some new step in the process ("thin" because I was reducing the number of nodes in the neurons)
Some other tags might be 50L (an RNN with 50 LSTM nodes), or 50D (a Dense layer with 50 nodes). If it end in -2 or -3, it means I was predicting either for two classes (ICU, Not ICU) or three classes (ICU, Hospitalized, Self Isolation).
This is one of those moments that makes it clear how important it is to organize your file structure in the moment... apologies for the mess.

---

Given the original dataset (Unstructured notes including PID and a separate Outcome file also including PID), the following is the order of the scripts to be run to turn the original notes data into unbiased GloVe vectors for training
-- Note; the filenames within the scripts are not automatically configured. Verify the filenames pointed to in each script before running to making sure you're reading from- and writing to- the file you intend to.

1. remove_unused_columns.py
2. merge_notes_and_outcomes.py
3. drop_biased_rows.py
4. preprocess_notes.py
5. cast_notes_to_glove.py
