import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from scipy.special import softmax

def accuracy(labels, predictions):
    return accuracy_score(labels, predictions)

def AUROC(labels, logits):
    probabilities = softmax(logits)

    # Assuming you are interested in the probability of the positive class
    flat_probabilities = probabilities[:, 1]

    return roc_auc_score(labels, flat_probabilities)

def precision(labels, predictions):
    return precision_score(labels, predictions)

def recall(labels, predictions):
    return recall_score(labels, predictions)

path = "DualESM/out/esm1b/"
labels = np.load(path + "labels.npy")
logits = np.load(path + "logits.npy")
predictions = np.load(path + "predictions.npy")

print("accuracy:")
print(accuracy(labels, predictions))

print("AUROC:")
print(AUROC(labels, logits))

print("precision:")
print(precision(labels, predictions))

print("recall:")
print(recall(labels, predictions))