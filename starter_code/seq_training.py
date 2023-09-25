import torch
import datasets
from datasets import load_dataset
import transformers
from torch.utils.data import Dataset
from datasets import load_metric
import numpy as np

def tokenize_data(data):
    return tokenizer(data['sequence'], padding='max_length')

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
    TrainingArguments
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
train_dataset = dataset["train"].shuffle(seed=10)
test_dataset = dataset["test"].shuffle(seed=10)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    fp16=False,
    save_strategy="no",
    weight_decay=0.015,
    output_dir="output"
)


class NewTrainer(Trainer):
    def establish_loss_weights(self):
        loss_vals = [
            (float(1184)/2732),  # 0 (negative)
            1 - (float(1184)/2732),  # 1 (positive)
            0.0,  # 3 padding token
        ]
        
        self.loss_weights = torch.Tensor(loss_vals).cuda()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs["logits"].cuda().float()

        labels_tensor = torch.tensor(inputs["labels"]).float().cuda()
        labels_tensor = torch.transpose(labels_tensor, 1, 2)

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.loss_weights)
        loss = loss_fct(logits, labels_tensor)
        return (loss, outputs) if return_outputs else loss

trainer = NewTrainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
    compute_metrics=compute_metrics
    # eval_steps
)

trainer.establish_loss_weights()

_ = trainer.train()

# evaluate
trainer.evaluate()
