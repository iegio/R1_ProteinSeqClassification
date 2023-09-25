import numpy as np
import pandas as pd

# print(np.load("tmp/labels.npy"))
# print(np.load("tmp/logits.npy"))
# print(np.load("tmp/predictions.npy"))

df = pd.read_csv("data/training_data.csv")
print(df["label"].sum())
