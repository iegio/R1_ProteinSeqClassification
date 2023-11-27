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


def tokenize_data(data):
    return tokenizer(data["sequence"], padding="max_length", max_length=512)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    np.save("out/genesplit/logits", logits)
    np.save("out/genesplit/labels", labels)
    np.save("out/genesplit/predictions", predictions)

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
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    fp16=False,
    save_strategy="no",
    weight_decay=0.015,
    output_dir="out"
)

class NewTrainer(Trainer):
    def establish_loss_weights(self, positive_pct):
        loss_vals = [
            positive_pct,  # 0 (negative)
            1 - positive_pct,  # 1 (positive)
        ]
        
        self.loss_weights = torch.Tensor(loss_vals).cuda()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss
        print(logits.shape)
        print(labels.shape)
        print(logits.view(-1, self.model.config.num_labels).shape)

        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.loss_weights, device=model.device))
        loss = loss_fct(logits.view(4, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = NewTrainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=df_train,  # training dataset
    eval_dataset=df_test,  # evaluation dataset
    compute_metrics=compute_metrics
)

positive_pct = float(np.sum(df_train["label"])) / len(df_train["label"])
trainer.establish_loss_weights(positive_pct)

_ = trainer.train()

trainer.evaluate()