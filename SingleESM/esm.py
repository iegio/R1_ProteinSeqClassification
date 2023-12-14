import torch
import datasets
from datasets import load_dataset
import transformers
from torch.utils.data import Dataset, DataLoader
from datasets import load_metric
import numpy as np
from torch import nn
import time
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


metric = load_metric("accuracy")

def tokenize_data(data):
    return tokenizer(data["sequence"], padding="max_length", max_length=512)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    np.save("out/esm1b_balanced/logits", logits)
    np.save("out/esm1b_balanced/labels", labels)
    np.save("out/esm1b_balanced/predictions", predictions)

    return metric.compute(predictions=predictions, references=labels)

from transformers import (
    EsmForMaskedLM, 
    BertTokenizer, 
    EsmTokenizer,
    EsmForSequenceClassification,
    EsmModel,
    pipeline,
    BertForMaskedLM,
    TrainingArguments,
    AutoTokenizer,
    Trainer
)

# model_name = "facebook/esm2_t36_3B_UR50D"
model_name = "facebook/esm1b_t33_650M_UR50S"

tokenizer = EsmTokenizer.from_pretrained(model_name, do_lower_case=False)
model = EsmForSequenceClassification.from_pretrained(model_name, num_labels = 2)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)

# load datasets
genesplit = False
my_path = ""
if(genesplit):
    my_path = "../data/genesplit_balanced/single/"
else:
    my_path = "../data/mixed_balanced/single/"

train_path = my_path + "train.csv"
test_path = my_path + "test.csv"
dataset = load_dataset('csv', data_files={'train': train_path, 'test': test_path})

# tokenize
dataset = dataset.map(tokenize_data, batched=True)

# remove columns
dataset = dataset.remove_columns(["Unnamed: 0", "identifier", "mut", "gene"])
# split
df_train = dataset["train"].shuffle(seed=10)
df_test = dataset["test"].shuffle(seed=10)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    fp16=False,
    save_strategy="no",
    weight_decay=0.015,
    output_dir="output",
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=df_train,  # training dataset
    eval_dataset=df_test,  # evaluation dataset
    compute_metrics=compute_metrics
)

_ = trainer.train()

# evaluate
trainer.evaluate()
