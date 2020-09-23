import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# These are for use for string eval
from numpy import array
from numpy import float32

# Constants
KFOLD_SPLITS = 10 

timeseries_df = pd.read_csv('./Covid_Data/Timeseries_Glove_Notes.csv')

# For simplicity, first we'll use the avg column as the input data
X_raw = timeseries_df['Notes in GloVe'].values
Y = timeseries_df['Admission Status'].values

# GloVe Avg column is storing arrays as strings, so they need to be unpacked
# Also pads X so all values are the same length
X = []
maxlen = 0
for x in X_raw:
    npx = np.array(eval(x))
    if npx.shape[0] > maxlen: maxlen = npx.shape[0]
    X.append(npx)

# Encode the labels to numerical representation
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
## encoded_Y or onehot_y?
## encoded_Y is an integer representation, which may (or may not)
## convey a false relationship between outcomes to the model. Of the categories
## Self Isolation -> Hospitalized -> ICU, there seems to be a viable relationship
## of outcome severity, so onehot may be counterprodutive
onehot_Y = np_utils.to_categorical(encoded_Y)

lstm = nn.LSTM(50, 3)


for sequence in X:
    sequence = torch.tensor(sequence)
    # can opt to randomly initialize hidden state here

