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
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

def tokenize_data(data):
    return tokenizer(data["sequence"], padding="max_length", max_length=512)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    probabilites = torch.nn.functional.softmax(torch.tensor(np.array(logits)), dim=1).cpu().detach().numpy()

    prob_of_positive = probabilites[:, 1]

    auroc = roc_auc_score(labels, prob_of_positive)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0,1]).ravel()

    total = np.sum([tn, fp, fn, tp])
    print("Accuracy : ",  (tp + tn) / total)
    print("Precision : ",  tp / (tp + fp))
    print("Recall : ",  tp / (tp + fn))
    print("AUROC : ",  auroc)

    # np.save("out/bert_balanced/logits", logits)
    # np.save("out/bert_balanced/labels", labels)
    # np.save("out/bert_balanced/predictions", predictions)

    return metric.compute(predictions=predictions, references=labels)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertTokenizer
)

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")

model = AutoModelForSequenceClassification.from_pretrained(
    "Rostlab/prot_bert_bfd",
     num_labels = 2
)

lora = False

if(lora):
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # trainable params: 2,308,098 || all params: 421,901,316 || trainable%: 0.5470705855774103

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# load datasets
genesplit = True
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
    output_dir="out"
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=df_train,  # training dataset
    eval_dataset=df_test,  # evaluation dataset
    compute_metrics=compute_metrics
)

_ = trainer.train()

trainer.evaluate()
