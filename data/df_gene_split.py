import pandas as pd
import numpy as np
import random
import math

random.seed(123)

def split_df(df, min_thresh, max_thresh):
    gene_names_original = df["gene"].unique()

    curr_length = df.shape[0]
    selected_genes = []

    # make sure we don't put too many in the test dataset
    while(curr_length > max_thresh * df.shape[0]):
        curr_length = 0
        selected_genes = []
        gene_names = gene_names_original.tolist()

        # make sure we have enough sequences in the test dataset
        while(curr_length < min_thresh * df.shape[0]):
            # select random gene
            my_gene = random.choice(gene_names)

            # count sequences matching that gene
            curr_length += df[df["gene"] == my_gene].shape[0]

            # record gene
            selected_genes.append(my_gene)
            gene_names.remove(my_gene)
    
    test_df = df[df["gene"].isin(selected_genes)]
    train_df = df.drop(test_df.index)
    
    return train_df, test_df


df = pd.concat([pd.read_csv("mixed_balanced/single/test.csv", index_col = 0), pd.read_csv("mixed_balanced/single/train.csv", index_col = 0)], ignore_index=True)

train_len = 0.0
total_len = df.shape[0]
min_thresh = 0.15
max_thresh = 0.2

train_df, test_df = split_df(df, min_thresh, max_thresh)

print(float(test_df.shape[0])/(test_df.shape[0] + train_df.shape[0]))

test_df.to_csv("genesplit_balanced/single/test.csv")
train_df.to_csv("genesplit_balanced/single/train.csv")
