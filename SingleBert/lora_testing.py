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


lr = 1e-3
batch_size = 16
num_epochs = 10 

def tokenize_data(data):
    return tokenizer(data["sequence"], padding="max_length", max_length=512)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    np.save("out/lora/logits", logits)
    np.save("out/lora/labels", labels)
    np.save("out/lora/predictions", predictions)

    return metric.compute(predictions=predictions, references=labels)

from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")

model = AutoModelForSequenceClassification.from_pretrained(
    "Rostlab/prot_bert_bfd",
     num_labels = 2
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=8, lora_dropout=0.1, bias="all"
)

model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()
# trainable params: 2,308,098 || all params: 421,901,316 || trainable%: 0.5470705855774103

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# load datasets
genesplit = False
my_path = ""
if(genesplit):
    my_path = "../data/genesplit/single/"
else:
    my_path = "../data/mixed/single/"

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
    output_dir="out",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="no",
    save_strategy="no",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=df_train,  # training dataset
    eval_dataset=df_test,  # evaluation dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

_ = trainer.train()

trainer.evaluate()
