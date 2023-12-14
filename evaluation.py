import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

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

path = "SingleESM/out/esm1b_balanced/"

labels = np.load(path + "labels.npy")
logits = np.load(path + "logits.npy")
predictions = np.load(path + "predictions.npy")

print("labels")
print(labels)
print("predictions")
print(predictions)
print("logits")
print(logits)

probabilites = torch.nn.functional.softmax(
    torch.tensor(np.array(logits)), 
    dim=1
).cpu().detach().numpy()

prob_of_positive = probabilites[:, 1]

auroc = roc_auc_score(labels, prob_of_positive)

tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0,1]).ravel()

total = np.sum([tn, fp, fn, tp])
print("Accuracy : ",  (tp + tn) / total)
print("Precision : ",  tp / (tp + fp))
print("Recall : ",  tp / (tp + fn))
print("AUROC : ",  auroc)

# print("Accuracy:")
# print(accuracy(labels, predictions))

# print("Precision:")
# print(precision(labels, predictions))

# print("Recall:")
# print(recall(labels, predictions))

# print("AUROC:")
# print(AUROC(labels, logits))
