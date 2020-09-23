import pandas as pd
import numpy as np
from keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.layers import SimpleRNN, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.utils import class_weight

from custom_classification_report import classification_report as class_report

# These are for use for string eval
from numpy import array
from numpy import float32

# Constants
KFOLD_SPLITS = 5

timeseries_df = pd.read_pickle('./Covid_Data/Bucket_Timeseries.pkl')

# For simplicity, first we'll use the avg column as the input data
X_raw = timeseries_df['Bucket Timeseries'].values
Y = timeseries_df['Admission Status'].values


## Reduce problem size to HOSPITAL / NOT HOSPITAL
for i in range(len(Y)):
    if Y[i] != "ICU":
        Y[i] = "Not ICU"

''' Currenty class breakdown:
    Total: 392 || Hospitalized: 221 | ICU: 143 | Self Isolation: 28 ||
    Removing some random Not ICU rows to equalize the data

    Find indices of all Not ICU rows, Shuffle
    Remove the first <Not ICU.count - ICU.count> from X and Y
'''
Y = pd.Series(Y)
amount_to_remove = Y.value_counts()['Not ICU'] - Y.value_counts()['ICU']
not_icu_indices = np.flatnonzero(Y == 'Not ICU')
np.random.shuffle(not_icu_indices)
indices_to_remove = not_icu_indices[:amount_to_remove]
# Drop the rows
Y = Y.drop(indices_to_remove).to_numpy()

print(X_raw)

# Pull X into a 3 dimensional array and drop unused indices
X = []
for i in range(len(X_raw)):
    if i not in indices_to_remove:
        x = np.array(X_raw[i])
        X.append(x)
X = np.array(X)

# Encode the labels to numerical representation
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

def sequential_model():
    # Intiiate RMSProp optimizer
    opt = Adam(lr=0.01, decay=1e-6)
    
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(None, 50), kernel_initializer='random_normal'))
    model.add(LeakyReLU())
    #model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    # Compile
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    return model


OUTPUT_FOLDER = "./training_output/"
FILE_LABEL = "RNN-50L-50D-2D"

report_df = pd.DataFrame()
current_fold = 0
def classification_report(y_true, y_pred):
    global report_df
    global current_fold

    fold_report = class_report(y_true, y_pred, encoder)
    fold_report['Fold'] = [current_fold] * len(fold_report)
    report_df = report_df.append(fold_report, ignore_index=True)
    report_df.to_csv(OUTPUT_FOLDER + FILE_LABEL + '.report.csv')
    
    current_fold += 1
    print(fold_report)

    return accuracy_score(y_true, y_pred)

# Instantiate callbcks
checkpoint_out = OUTPUT_FOLDER + FILE_LABEL + "_fold" + str(current_fold) + ".best.h5"
checkpoint = ModelCheckpoint(checkpoint_out, monitor='loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='loss', patience=20)

# Train
classifier = KerasClassifier(build_fn=sequential_model, epochs=500, batch_size=50)
kfold = KFold(n_splits=KFOLD_SPLITS)
results = cross_val_score(classifier, X, encoded_Y,
        cv=kfold, scoring=make_scorer(classification_report),
        fit_params={'callbacks': [checkpoint, early_stopping]}
    )
print("Baseline: %.2f%% (stdev %.2f%%)" % (results.mean()*100, results.std()*100))

# Display and save results
print(report_df)
report_df.to_csv(OUTPUT_FOLDER + FILE_LABEL + '.report.csv')

### Auxiliary Code ###

## Calculate weights for each class
##weights = class_weight.compute_class_weight('balanced', np.unique(encoded_Y), encoded_Y)
##print("WEIGHTS")
##print(weights)
#
## Create a loss function that takes weights into account
#def weighted_categorical_crossentropy(weights):
#    ## From Github: Wassname
#    """
#    A weighted version of keras.objectives.categorical_crossentropy
#
#    Variables:
#        weights: numpy array of shape (C,) where C is the number of classes
#    """
#    weights = K.variable(weights)
#
#    def loss(y_true, y_pred):
#        # scale predictions so that the class probas of each sample sum to 1
#        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#        # clip to prevent NaN's and Inf's
#        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#        # Calc
#        loss = y_true * K.log(y_pred) * weights
#        loss = -K.sum(loss, -1)
#        return loss
#
#    return loss
#
## A separate loss function to try
#def categorical_focal_loss(gamma=2.0, alpha=0.25):
#    ## From Github: aldi-dimara
#    """
#    Implementation of FOcal Loss from the paper in multiclas classification
#    Formula:
#        loss = -alpha*((1-p)^gamma)*log(p)
#    Parameters:
#        alpha -- the same as wegithing factor in balanced crossentropy
#        gamma -- focusing parameter for modulating factor (1-p)
#    Default value:
#        gamma -- 2.0 as mentioned in the paper
#        alpha -- 0.25 as mentioned in the paper
#    """
#    def focal_loss(y_true, y_pred):
#        # Define epsilon so that backpropagation will not result in NaN
#        # for 0 divisor case
#        epsilon = K.epsilon()
#        # Add the epsilon to prediction value
#        #y_pred = y_pred + epsilon
#        # Clip the prediction value
#        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
#        # Calculate crossentropy
#        cross_entropy = -y_true * K.log(y_pred)
#        # Calculate weight that consists of modulating factor an weighting factor
#        weight = alpha * y_true * K.pow((1-y_pred), gamma)
#        # Calculate focal loss
#        loss = weight * cross_entropy
#        # Sum the losses in mini-batch
#        loss = K.sum(loss, axis=1)
#        return loss
#
#    return focal_loss
#

