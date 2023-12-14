import pandas as pd
import numpy as np

df = pd.concat([pd.read_csv("mixed/dual/test.csv", index_col = 0), pd.read_csv("mixed/dual/train.csv", index_col = 0)], ignore_index=True)

df_2 = pd.read_csv("raw_data/DoCM_variants_v3.2.tsv", sep="\t")

df1_genes = df["gene"].unique()
df1_genes = [gene.split("_")[0] for gene in df1_genes]
df2_genes = df_2["gene"].unique()


# print(df1_genes)
# print(df2_genes)

my_intersection = [value for value in df2_genes if value in df1_genes]
# print(0/0)
# df[df["gene"].isin(selected_genes)]
my_dfs = []
for gene in df["gene"].unique():
    df_gene = df[df["gene"] == gene]
    df_pos = df_gene[df_gene["label"] == 1]
    df_neg = df_gene[df_gene["label"] == 0]

    if(gene.split("_")[0] in my_intersection and len(df_neg) > len(df_pos)):
        # add positive examples into df_pos
        my_examples = list(df_2[df_2["gene"] == gene.split("_")[0]]["amino_acid"])
        for example in my_examples:
            wt = example[2]
            mut = example[-1]
            pos = example[3:-1]

            # if wt matches, append row to df_neg
            my_index = int(pos)*2 - 2
            if(wt == list(df_gene["wt"])[0][my_index]):
                new_data = pd.DataFrame(df_neg[-1:].values, columns=df_neg.columns)
                mut_seq = list(df_gene["wt"])[0]
                mut_seq = mut_seq[:my_index] + mut + mut_seq[my_index + 1:]
                new_data.loc[0, "mut"] = mut_seq
                new_data.loc[0, "variant"] = str(wt + pos + mut)
                new_data.loc[0, "label"] = 1
                df_pos = pd.concat([df_pos, new_data], ignore_index=True)

    # else:
    my_min = min(len(df_pos), len(df_neg))

    df_pos = df_pos.sample(my_min, random_state=123)
    df_neg = df_neg.sample(my_min, random_state=123)

    my_dfs.append(df_pos)
    my_dfs.append(df_neg)

result = pd.concat(my_dfs)

# randomly subsample rows for training/testing
result = result.sample(frac=1, random_state=123).reset_index(drop=True)

print(len(result))
print(len(result["gene"].unique()))
testing_df = result.copy().sample(frac = 0.2, replace=False, random_state=123)
training_df = result.drop(testing_df.index)

testing_df.to_csv("mixed_balanced/dual/test.csv")
training_df.to_csv("mixed_balanced/dual/train.csv")

# dual
# ,identifier,label,variant,wt,mut,gene
# ,identifier,label,mut,sequence,gene

# single
# ,identifier,mut,label,sequence,gene

# "variant" becomes "mut"
# "mut" becomes "sequence"
# delete "wt"

testing_df_single = testing_df.drop("wt", axis='columns')
training_df_single = training_df.drop("wt", axis='columns')

testing_df_single.rename(columns={"mut": "sequence", "variant": "mut"})
training_df_single.rename(columns={"mut": "sequence", "variant": "mut"})

testing_df_single.to_csv("mixed_balanced/single/test.csv")
training_df_single.to_csv("mixed_balanced/single/train.csv")

