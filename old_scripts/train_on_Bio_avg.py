import pandas as pd
import numpy as np
from keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.utils import class_weight

from custom_classification_report import classification_report

# Constants
KFOLD_SPLITS = 10 

gloved_csv = pd.read_csv('./Covid_Data/Concatted_Notes_BioWordVec_averages_200d.csv')

# For simplicity, first we'll use the avg column as the input data
X_raw = gloved_csv['BioWordVec Avg'].values
Y = gloved_csv['Admission Status'].values

# GloVe Avg column is storing arrays as strings, so they need to be unpacked
X = []
indices_to_drop = []
def array_from_str(instr):
    return (np.fromstring(instr[1:-1], sep=' ')).tolist()
for i in range(len(X_raw)):
    X.append(array_from_str(X_raw[i]))
X = np.array(X)

# Encode the labels to numerical representation
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

def baseline_model():
    # 50 inputs -> [64 hidden nodes] -> 3 outputs
    model = Sequential()
    model.add(Dense(100, input_dim=200, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

report_df = pd.DataFrame()
current_fold = 0
def classification_report_w_acc_score(y_true, y_pred):
    global report_df
    global current_fold

    fold_report = classification_report(y_true, y_pred, encoder)
    fold_report['Fold'] = [current_fold] * len(fold_report)
    report_df = report_df.append(fold_report, ignore_index=True)
    
    current_fold += 1
    print(fold_report)
    
    return accuracy_score(y_true, y_pred)

# Callbacks
early_stopping = EarlyStopping(monitor='loss', patience=10)

classifier = KerasClassifier(build_fn=baseline_model, epochs=400, batch_size=5)
# KFold cross-validation: partition into k bins and use k-1 as training data, 1 as testing
kfold = KFold(n_splits=KFOLD_SPLITS)
# Fit and evaluate model
results = cross_val_score(classifier, X, encoded_Y,
        cv=kfold, scoring=make_scorer(classification_report_w_acc_score),
        fit_params={'callbacks': [early_stopping]}
    )
print("Baseline: %.2f%% (stdev %.2f%%)" % (results.mean()*100, results.std()*100))
# Display and save results
print(report_df)
REPORT_FILE = "200d_biovec_100-3.csv"
report_df.to_csv("./classification_reports/" + REPORT_FILE)

# --- Results on non-concatted notes --- #
## 50 inputs -> [64 hidden nodes] -> 3 outputs
# Testing accuracy on first run: 55.41$ (5.07% stddev)
## 50 inputs -> [ 40 hn -> 30 hn ] -> 3 outputs
# Testing accuracy: 54.87% (5.59% stdev)

# --- Averages of Concatted notes (Sans Self Isolation) --- #
## 50 inputs -> [50 hn] -> 2 outputs
# Testing accuracy: 82.70% (4.42% stddev)

# --- Averaged of Concatted notes (With Self Isolation) --- #
## 50 inputs -> [50 hn] -> 3 outputs
# Testing Accuracy: 81.14% (4.93% stddev)
## 50 input -> [30 hn -> 20 hn] -> 3 output
# Testing Accuracy: 79.34% (5.29% stddev)

# 400 epoch
## 50 inputs -> [40hn -> 15hn] -> 3 oututs
# Testing Accuracy: 77.06% (stdev 7.58%)
