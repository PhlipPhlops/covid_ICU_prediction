import pprint
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def score_counter(val, y_true, y_pred):
    scores = {
            "TP": 0,
            "FP": 0,
            "TN": 0,
            "FN": 0
        }
    
    for i in range(len(y_pred)):
        if y_pred[i] == val:
            # positives
            if y_true[i] == val:
                scores["TP"] += 1
            else:
                scores["FP"] += 1
        else:
            # negatives
            if y_true[i] == val:
                scores["FN"] += 1
            else:
                scores["TN"] += 1

    for key in scores.keys():
        scores[key] = float(scores[key])

    return scores

def accuracy(scores):
    correct = scores["TP"] + scores["TN"]
    incorrect = scores["FP"] + scores["FN"]
    return correct / (correct + incorrect)

def specificity(scores):
    # TP / TP + FN
    if scores["TP"] == 0: return 0
    return scores["TP"] / (scores["TP"] + scores["FN"])

def sensitivity(scores):
    # TN / FP + TN
    if scores["TN"] == 0: return 0
    return scores["TN"] / (scores["FP"] + scores["TN"])

def positive_predictive_value(scores):
    # TP / TP + FP
    if scores["TP"] == 0: return 0
    return scores["TP"] / (scores["TP"] + scores["FP"])

def negative_predictive_value(scores):
    # TN / TN + FN
    if scores["TN"] == 0: return 0
    return scores["TN"] / (scores["TN"] + scores["FN"])

def to_binary_labels(val, y):
    bin_copy = y.copy()
    for i in range(len(bin_copy)):
        if bin_copy[i] == val:
            bin_copy[i] = 1
        else:
            bin_copy[i] = 0
    return bin_copy

def classification_report(y_true, y_pred, labelEncoder):
    # Returns a dataframe representing a range of scores for each label
    classes = labelEncoder.classes_
    class_report = {
            "Class": [],
            "Accuracy": [],
            "Specificity": [],
            "Sensitivity": [],
            "PPV": [],
            "NPV": [],
            "ROC_AUC": [],
            "Average Precision Score": []
        }
    report_df = pd.DataFrame(data=class_report)
    for i in range(len(classes)):
        scores = score_counter(i, y_true, y_pred)
        print(scores)
        print(y_true, y_pred)
        row = {
                "Class": classes[i],
                "Accuracy": accuracy(scores),
                "Specificity": specificity(scores),
                "Sensitivity": sensitivity(scores),
                "PPV": positive_predictive_value(scores),
                "NPV": negative_predictive_value(scores)
            }
        # ROC AUC is undefined if all values are the same. So verify difference fist
        bin_y_true = to_binary_labels(i, y_true)
        bin_y_pred = to_binary_labels(i, y_pred)
        if len(set(bin_y_true)) > 1 and len(set(bin_y_pred)) > 1:
            row["ROC_AUC"] = roc_auc_score(bin_y_true, bin_y_pred)
            row["Average Precision Score"] = average_precision_score(bin_y_true, bin_y_pred)
        else:
            row["ROC_AUC"] = "undefined"
            row["Average Precision Score"] = "undefined"

        report_df = report_df.append(row, ignore_index=True)

    return report_df
