import pandas as pd
import numpy as np

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
traintestdf = pd.DataFrame(columns=["identifier", "variant", "gene", "wt", "mut", "label"])

my_count = 0

# go through each unique protein identifier
for i in label_identifiers:
    piece = labels.loc[labels["identifier"] == i]
    my_protein_seq = filtered_sequences.loc[filtered_sequences["identifier"] == i]["sequence"].iloc[0]
    my_protein_gene = filtered_sequences.loc[filtered_sequences["identifier"] == i]["gene"].iloc[0]

    # iterate through each mutation in the labels file
    for j in range(piece.shape[0]):
        
        row = piece.iloc[j]
        variant = row["location"]
        label = row["label"]

        # check if wt matches
        if(wt_match(variant, my_protein_seq)):
            new_protein_seq = swap_seq_location(variant, my_protein_seq)
            new_row = [i, variant, my_protein_gene, my_protein_seq, new_protein_seq, label]

            # append new row
            traintestdf.loc[len(traintestdf.index)] = new_row
        my_count += 1
            
print("matched: " + str(counter_matched))
print("unmatched: " + str(counter_unmatched))

# eliminate proteins with > 500 AA
max_len = 512
print("samples before eliminating proteins with >500 AA: ", traintestdf.shape[0])
traintestdf = traintestdf[traintestdf['mut'].str.len() <= max_len]
print("samples after eliminating proteins with >500 AA: ", traintestdf.shape[0])

# randomly subsample rows for training/testing
testing_df = traintestdf.copy().sample(frac = 0.1, replace=False, random_state = 123)
training_df = traintestdf.drop(testing_df.index)

# reformat sequences
testing_df["wt"] = split_seqs(testing_df["wt"])
testing_df["mut"] = split_seqs(testing_df["mut"])

training_df["wt"] = split_seqs(training_df["wt"])
training_df["mut"] = split_seqs(training_df["mut"])

testing_df.to_csv("data/test_df.csv")
training_df.to_csv("data/train_df.csv")
