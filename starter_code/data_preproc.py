import pandas as pd
import numpy as np

import sys

counter_matched = 0
counter_unmatched = 0

def split_seqs(seq_list):
    return [' '.join(AA for AA in seq) for seq in seq_list]

def wt_match(mut_loc, protein_seq):
    global counter_matched
    global counter_unmatched
    wt = mut_loc[0]
    loc = int(mut_loc[1:-1]) -1

    if(len(protein_seq) > loc and protein_seq[loc] == wt):
        counter_matched += 1
        return True
    
    counter_unmatched += 1
    return False

def swap_seq_location(mut_loc, protein_seq):
    loc = int(mut_loc[1:-1]) - 1
    mut = mut_loc[-1]

    protein_seq = protein_seq[:loc] + mut + protein_seq[loc + 1:]

    return protein_seq

sequences = pd.read_csv("data/all_DRGN_seqs.csv", index_col=0)
labels = pd.read_csv("data/DRGN.txt", sep="\t", header=None, names=["identifier", "location", "label"])

# extract the rows we care about
label_identifiers = labels["identifier"].unique()
filtered_sequences = sequences.loc[sequences['identifier'].isin(label_identifiers), :]

# create a new, empty dataframe for us to use for training/testing
traintestdf = pd.DataFrame(columns=["identifier", "mut", "label", "sequence", "gene"])

my_count = 0
# go through each unique protein identifier
for i in label_identifiers:
    piece = labels.loc[labels["identifier"] == i]
    my_protein_seq = filtered_sequences.loc[filtered_sequences["identifier"] == i]["sequence"].iloc[0]
    my_protein_gene = filtered_sequences.loc[filtered_sequences["identifier"] == i]["gene"].iloc[0]

    # iterate through each mutation in the labels file
    for j in range(piece.shape[0]):
        
        row = piece.iloc[j]
        mut = row["location"]
        label = row["label"]

        # check if wt matches
        if(wt_match(mut, my_protein_seq)):
            new_protein_seq = swap_seq_location(mut, my_protein_seq)
            new_row = [i, mut, label, new_protein_seq, my_protein_gene]

            # append new row
            traintestdf.loc[len(traintestdf.index)] = new_row
        my_count += 1
            
print("matched: " + str(counter_matched))
print("unmatched: " + str(counter_unmatched))

# randomly subsample rows for training/testing
testing_df = traintestdf.copy().sample(frac = 0.1, replace=False)
training_df = traintestdf.drop(testing_df.index)

# reformat sequences
testing_df["sequence"] = split_seqs(testing_df["sequence"])
training_df["sequence"] = split_seqs(training_df["sequence"])

testing_df.to_csv("data/testing_data.csv")
training_df.to_csv("data/training_data.csv")
