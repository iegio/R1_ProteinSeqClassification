import numpy as np
import pandas as pd

def accuracy(labels, predictions):
    return float(np.sum(labels == predictions))/len(labels)

def AUROC(labels, predictions):
    return 0

def precision(labels, predictions):
    return 0

def recall(labels, predictions):
    return 0

labels = np.load("tmp/original_weights/labels.npy")
predictions = np.load("tmp/original_weights/predictions.npy")

print("accuracy:")
print(accuracy(labels, predictions))
