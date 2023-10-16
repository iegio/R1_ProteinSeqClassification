import torch
import datasets
from datasets import load_dataset
import transformers
from torch.utils.data import Dataset
from datasets import load_metric
import numpy as np
from torch import nn

def split_seqs(seq_list):
    return [' '.join(AA for AA in seq) for seq in seq_list]

def tokenize_data(data):
    return tokenizer(data["sequence"], padding="max_length", max_length=512)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    np.save("tmp/logits", logits)
    np.save("tmp/labels", labels)
    np.save("tmp/predictions", predictions)

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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# load datasets
dataset = load_dataset('csv', data_files={'train': 'data/training_data.csv', 'test': 'data/testing_data.csv'})

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
    output_dir="output"
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
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.loss_weights, device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=df_train,  # training dataset
    eval_dataset=df_test,  # evaluation dataset
    compute_metrics=compute_metrics
)

# positive_pct = float(np.sum(df_train["label"])) / len(df_train["label"])
# trainer.establish_loss_weights(positive_pct)

_ = trainer.train()

# evaluate
trainer.evaluate()
