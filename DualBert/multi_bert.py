from transformers import ( 
    AutoConfig,
    BertModel, 
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

import pandas as pd
import numpy as np

import torch 
from torch.utils.data import DataLoader, Dataset 
import torch.nn.functional as F
import torch.nn as nn 
from torch.optim import AdamW
import copy
from datasets import load_dataset
 
def tokenize_data(data):
    tokens1 = data["wt"].apply(lambda x: tokenizer(x, padding="max_length", max_length=512))
    tokens2 = data["mut"].apply(lambda x: tokenizer(x, padding="max_length", max_length=512))
    return tokens1, tokens2

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    np.save("out/logits", logits)
    np.save("out/labels", labels)
    np.save("out/predictions", predictions)

    return metric.compute(predictions=predictions, references=labels)

class DualBertForClassification(nn.Module):
    def __init__(self, bert_model_a, bert_model_b):
        super(DualBertForClassification, self).__init__()
        
        self.bert_model_wt = bert_model_a
        self.bert_model_mutant = bert_model_b

        # wildtype_hidden_states = bert_model_a.last_hidden_state
        # mutant_hidden_states = bert_model_b.last_hidden_state

        # concatenated_hidden_states = torch.cat([wildtype_hidden_states, mutant_hidden_states], dim=-1)

        # nn.Linear(in_features=1024, out_features=2, bias=True)


        # Define the feed-forward layer
        self.linear1 = nn.Linear(2048, 2048)
        # self.classifier = nn.Sequential(
        #     ,
        # )

        # self.linear1 = nn.Linear(...)
        # self.linear2 = nn.Linear(...)

        ##############################
        
        # TODO : attach the clasification head 
        #  (pooler): BertPooler(
        #     (dense): Linear(in_features=1024, out_features=1024, bias=True)
        #     (activation): Tanh()
        #     )
        # )
        # (dropout): Dropout(p=0.0, inplace=False)
        # (classifier): Linear(in_features=1024, out_features=2, bias=True)

        ##############################
        
        ### This is an example of transforming from this  to [batch_sz, 1], 
        ### Ignore Below this vv
        ### FOR EXAMPLE ONLY, DELETE ALL OF THIS BELOW 
        # self.layer_1 = nn.Linear(1024, 512)
        # self.layer_2 = nn.Linear(512, 128)
        # self.layer_3 = nn.Linear(128, 16)
        # self.layer_4 = nn.Linear(16, 1)

    def forward(self, x):
        # wild type input, mutant input
        (x_a, x_b) = x

        # concatenate
        x = torch.cat(
            (
                self.bert_model_wt(**x_a), # [batch_sz, sequence_sz, 1024]
                self.bert_model_mutant(**x_b) # [batch_sz, sequence_sz, 1024]
            ), 
            1
        )  # [batch_sz, 2*sequence_sz, 1024]

        x = torch.tanh(self.linear1(x))

        # x = self.linear_1(in_features=1024, out_features=1024, bias=True)
        # x = 0

        ##############################

        # TODO : make the forward pass with the new classififcation head
        #  (pooler): BertPooler(
        #     (dense): Linear(in_features=1024, out_features=1024, bias=True)
        #     (activation): Tanh()
        #     )
        # )
        # (dropout): Dropout(p=0.0, inplace=False)
        # (classifier): Linear(in_features=1024, out_features=2, bias=True)
        
        
        ##############################

        ### This is an example of transforming from this  to [batch_sz, 1], 
        ### FOR EXAMPLE ONLY, DELETE ALL OF THIS BELOW 
        # x = torch.tanh(self.layer_1(x)) # [batch_sz, 2*sequence_sz, 512, ]
        # x = torch.tanh(self.layer_2(x)) # [batch_sz, 2*sequence_sz, 128, ]
        # x = torch.tanh(self.layer_3(x)) # [batch_sz, 2*sequence_sz, 16, ]
        # x = torch.tanh(self.layer_4(x)) # [batch_sz, 2*sequence_sz, 1, ]
        # x = torch.mean(x, dim = 1)  # [batch_sz, 1]
        return x
    

class DualSequenceDataset(Dataset):
    def __init__(self, tokenized_inputs_wt, tokenized_inputs_mut, sequence_labels):
        self.sequences_wt = tokenized_inputs_wt
        self.sequences_mut = tokenized_inputs_mut
        self.labels = sequence_labels

    def __len__(self):
        return len(self.labels)

    # output is a dictionary, not a tuple
    def __getitem__(self, idx):
        print(self.sequences_wt[idx])
        item = {"x": (self.sequences_wt[idx].data, self.sequences_mut[idx].data)}
        item['label'] = torch.tensor(self.labels[idx])
        return item

        # seq_pair = (self.sequences_wt[idx], self.sequences_mut[idx])
        # label = self.labels[idx]
        # return seq_pair, label


def main():
    return


if __name__ == "__main__":

    # process data
    # sequences = pd.read_csv("./data/all_DRGN_seqs.csv", )
    # id_to_seq = dict(zip(sequences["identifier"].values, sequences["sequence"].values))
    # variants = pd.read_csv("./data/DRGN.txt", names = ["protein", "variant", "label"], delimiter = "\t")
    model_name = "Rostlab/prot_bert_bfd"
    
    tokenizer = BertTokenizer.from_pretrained(model_name,  do_lower_case = False)
    
    model1 = BertModel.from_pretrained(model_name)
    model2 = BertModel.from_pretrained(model_name)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = DualBertForClassification(model1, model2).to(device)

    ##############################

    # TODO : use the Trainer class like you've been doing 

    ##############################

    # load datasets
    train_df = pd.read_csv("data/train_df.csv")
    test_df = pd.read_csv("data/test_df.csv")

    # tokenize
    train_df["wt"], train_df["mut"] = tokenize_data(train_df)
    test_df["wt"], test_df["mut"] = tokenize_data(test_df)

    train_ds = DualSequenceDataset(train_df["wt"].to_list(), train_df["mut"].to_list(), train_df["label"].to_list())
    test_ds = DualSequenceDataset(test_df["wt"].to_list(), test_df["mut"].to_list(), test_df["label"].to_list())
    # dl_train = DataLoader(train_ds, batch_sz = 2, shufflle = False)
    # dl_test = DataLoader(test_ds, batch_sz = 2, shufflle = False)

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        fp16=False,
        save_strategy="no",
        weight_decay=0.015,
        output_dir="out"
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_ds,  # training dataset
        eval_dataset=test_ds,  # evaluation dataset
        compute_metrics=compute_metrics
    )  

    _ = trainer.train()
