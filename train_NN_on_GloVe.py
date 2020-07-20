import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Constants
KFOLD_SPLITS = 10 

gloved_csv = pd.read_csv('./Covid_Data/Notes_w_GloVe_and_Outcome.csv')

# For simplicity, first we'll use the avg column as the input data
X_raw = gloved_csv['GloVe Avg'].values
Y = gloved_csv['Admission_status'].values

# GloVe Avg column is storing arrays as strings, so they need to be unpacked
X = []
indices_to_drop = []
def array_from_str(instr):
    return (np.fromstring(instr[1:-1], sep=' ')).tolist()
for i in range(len(X_raw)):
    if type(X_raw[i]) != str:
        indices_to_drop.append(i)
    else:
        X.append(array_from_str(X_raw[i]))
X = np.array(X)

# Deleting malformed data; these indices weren't added to X
print("Deleting {} rows of malformed data".format(len(indices_to_drop)))
Y = np.delete(Y, indices_to_drop)
print(X.shape)
print(Y.shape)

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

def baseline_model():
    # 50 inputs -> [64 hidden nodes] -> 3 outputs
    model = Sequential()
    model.add(Dense(40, input_dim=50, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=500)
# KFold cross-validation: partition into k bins and use k-1 as training data, 1 as testing
kfold = KFold(n_splits=KFOLD_SPLITS)
# Fit and evaluate model
results = cross_val_score(estimator, X, onehot_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

## 50 inputs -> [64 hidden nodes] -> 3 outputs
# Testing accuracy on first run: 55.41$ (5.07% stddev)
## 50 inputs -> [ 40 hn -> 30 hn ] -> 3 outputs
# Testing accuracy: 54.87% (5.59% stdev)
