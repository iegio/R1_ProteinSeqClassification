import pandas as pd
import numpy as np
import random
import math

def split_df(df, min_thresh, max_thresh):
    goal_length = min_thresh * df.shape[0]
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


df = pd.concat([pd.read_csv("data/testing_data.csv", index_col = 0), pd.read_csv("data/training_data.csv", index_col = 0)], ignore_index=True)

train_len = 0.0
total_len = df.shape[0]
min_thresh = 0.1
max_thresh = 0.2

train_df, test_df = split_df(df, min_thresh, max_thresh)

print(test_df.size)
print(train_df.size)

test_df.to_csv("data/genesplit_testing_data.csv")
train_df.to_csv("data/genesplit_training_data.csv")
