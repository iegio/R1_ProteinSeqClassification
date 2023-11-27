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
from sklearn.metrics import roc_auc_score

import torch 
from torch.utils.data import DataLoader, Dataset 
import torch.nn.functional as F
import torch.nn as nn 
from torch.optim import AdamW
import copy
from datasets import load_dataset
from datasets import load_metric
from sklearn.metrics import confusion_matrix
metric = load_metric("accuracy")

def tokenize_data(data, device="cuda"):
    tokens1 = data["wt"].apply(
        lambda x: tokenizer(x.split(" "), is_split_into_words=True,  return_tensors = "pt", padding="max_length", max_length=512).to(device)
    )

    tokens2 = data["mut"].apply(
        lambda x: tokenizer(x.split(" "), is_split_into_words=True, return_tensors = "pt", padding="max_length", max_length=512).to(device)
    )

    return tokens1, tokens2

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)

#     np.save("out/dualbert/logits", logits)
#     np.save("out/dualbert/labels", labels)
#     np.save("out/dualbert/predictions", predictions)

#     return metric.compute(predictions=predictions, references=labels)

class DualBertForClassification(nn.Module):
    def __init__(self, bert_model_a, bert_model_b):
        super(DualBertForClassification, self).__init__()
        self.num_labels = 2
        self.batch_size = 4

        self.bert_model_wt = bert_model_a
        self.bert_model_mutant = bert_model_b
        
        # Define the feed-forward layer
        self.linear1 = nn.Linear(2048, 1024)

        self.linear2 = nn.Linear(1024, 1024)

        self.linear3 = nn.Linear(512, self.num_labels)

    def forward(self, x):
        # wild type input, mutant input
        (x_a, x_b) = x
        
        x1 = self.bert_model_wt(**x_a).last_hidden_state
        x2 = self.bert_model_mutant(**x_b).last_hidden_state        
        x = torch.cat([x1, x2], 2)
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = torch.mean(x, dim = 2)
        x = torch.tanh(self.linear3(x))
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
        item = {"x": (self.sequences_wt[idx].data, self.sequences_mut[idx].data)}
        item['label'] = torch.tensor(self.labels[idx])
        return item
    

class MyTrainer(Trainer):

    def my_collate_fn(self, data_list):
        key_to_pad_token = {
            "token_type_ids" : TYPE_ID_PAD_TOKEN,
            "input_ids" : SEQUENCE_PAD_TOKEN,
            "attention_mask" : ATTENTION_MASK_PAD_TOKEN
        }
        lengths = [entry["x"][0]["input_ids"].shape[1] for entry in data_list]
        mx = np.max(lengths)
        y_list = [entry["label"] for entry in data_list]
        res = {0 : [], 1 : []}
        wt_inputs = [entry["x"][0] for entry in data_list]
        mut_inputs = [entry["x"][1] for entry in data_list]
        for j, data in enumerate([wt_inputs, mut_inputs]):
            key_to_dtype = {k : v.dtype for (k,v) in data[0].items()}
            key_to_values= {k : [] for k in data[0].keys()}
            for entry in data:
                for key, pad_token in key_to_pad_token.items():
                    req = mx - entry[key].shape[1]
                    if req == 0:
                        key_to_values[key].append(entry[key][0])  #### HUGE : might affect behavior if we do entry[key][0], padding[0] instaed of entry[key], padding
                    else:
                        padding = torch.ones(1, req) * pad_token
                        padded_tensor = torch.cat(
                            [entry[key][0], padding[0]], #### HUGE : might affect behavior if we do entry[key][0], padding[0] instaed of entry[key], padding
                            dim = 0
                        )
                        padded_tensor = padded_tensor.to(key_to_dtype[key])
                        key_to_values[key].append(padded_tensor)
            
                to_return = {k : torch.stack(v) for (k,v) in key_to_values.items()}
                res[j] = to_return
        
        # Evaluate loop assumes this returns a tuple, we had it as a dict before
        return (res[0], res[1]), torch.Tensor(y_list)
    
    def get_train_dataloader(self):
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=4, collate_fn=self.my_collate_fn) 
        return train_dl
    
    def get_test_dataloader(self):
        test_dl = DataLoader(test_ds, shuffle=True, batch_size=4, collate_fn=self.my_collate_fn) 
        return test_dl
    
    def compute_loss(self, model, inputs):
        (x, labels) = inputs # tuple : x is also a tuple (input_wt, input_mut), labels is a tensor  
        
        logits = model(x)

        loss_fct = nn.CrossEntropyLoss()
        labels = labels.to(torch.int64)
        loss = loss_fct(logits, labels)
        return loss

if __name__ == "__main__":
    global LABEL_PAD_TOKEN
    global ATTENTION_MASK_PAD_TOKEN
    global SEQUENCE_PAD_TOKEN
    global TYPE_ID_PAD_TOKEN
    global DOMAINS_PAD_TOKEN

    model_name = "Rostlab/prot_bert_bfd"
    
    tokenizer = BertTokenizer.from_pretrained(model_name,  do_lower_case = False, num_labels = 2)

    SEQUENCE_PAD_TOKEN = tokenizer.pad_token_id
    TYPE_ID_PAD_TOKEN = tokenizer.pad_token_type_id
    LABELS_PAD_TOKEN = -100
    ATTENTION_MASK_PAD_TOKEN = 0
    
    model1 = BertModel.from_pretrained(model_name)
    model2 = BertModel.from_pretrained(model_name)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = DualBertForClassification(model1, model2).to(device)

    # load datasets
    genesplit = False
    my_path = ""
    if(genesplit):
        my_path = "../data/genesplit/dual/"
    else:
        my_path = "../data/mixed/dual/"

    train_path = my_path + "train.csv"
    test_path = my_path + "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # tokenize
    train_df["wt"], train_df["mut"] = tokenize_data(train_df, device)
    test_df["wt"], test_df["mut"] = tokenize_data(test_df, device)

    train_ds = DualSequenceDataset(train_df["wt"].to_list(), train_df["mut"].to_list(), train_df["label"].to_list())
    test_ds = DualSequenceDataset(test_df["wt"].to_list(), test_df["mut"].to_list(), test_df["label"].to_list())

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        fp16=False,
        save_strategy="no",
        weight_decay=0.015,
        output_dir="out"
    )

    trainer = MyTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_ds,  # training dataset
        eval_dataset=test_ds,  # evaluation dataset
        # compute_metrics=compute_metrics
    )

    _ = trainer.train()


test_dl = trainer.get_test_dataloader()
with torch.no_grad():
    model.eval()

    labels = []
    logits = []
    for batch in test_dl:
        (inputs_1, inputs_2) = batch[0]
        batch_labels = batch[1].cpu()
        batch_predictions = model((inputs_1, inputs_2)).cpu().detach().numpy()
        labels.extend(batch_labels)
        logits.extend(batch_predictions)

    # do compute metrics here
    predictions = np.argmax(logits, axis=-1)

    np.save("out/dualbert/labels", labels)
    np.save("out/dualbert/logits", logits)
    np.save("out/dualbert/predictions", predictions)

    # converts logits to probablilities, won't affect our metrics below but 
    # may be useful in future cases
    
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

    # From James's commit: 
    # Accuracy :  0.7590361445783133
    # Precision :  0.7446808510638298
    # Recall :  0.813953488372093
    # AUROC :  0.8540697674418605

    # From Masha's commit:
    # Accuracy:
    # 0.7710843373493976
    # Precision:
    # 0.7727272727272727
    # Recall:
    # 0.7906976744186046
    # AUROC:
    # 0.8476744186046512
