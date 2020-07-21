import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
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

GLOVEd = 50
print("Padding values to consistent size (assuming GLOVEd={}). This may take a while...".format(GLOVEd))
for i in range(len(X)):
    if X[i].shape[0] == maxlen: continue
    X[i] = np.append(X[i], [[0.0] * GLOVEd] * (maxlen - X[i].shape[0]), axis=0)
X = np.array(X)
print("Done padding. Output shape is {}".format(X.shape))

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

def sequential_model():
    # 50 inputs -> [64 hidden nodes] -> 3 outputs
    model = Sequential()
    model.add(LSTM(50, input_shape=(maxlen, 50), return_sequences=False, activation='relu'))
    # model.add(Dense(15, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

estimator = KerasClassifier(build_fn=sequential_model, epochs=200, batch_size=5)
# KFold cross-validation: partition into k bins and use k-1 as training data, 1 as testing
kfold = KFold(n_splits=KFOLD_SPLITS)
# Fit and evaluate model
results = cross_val_score(estimator, X, onehot_Y, cv=kfold)
print("Baseline: %.2f%% (stdev %.2f%%)" % (results.mean()*100, results.std()*100))

