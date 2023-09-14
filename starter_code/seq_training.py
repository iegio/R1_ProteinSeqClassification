import torch
import datasets
from datasets import load_dataset
import transformers
# import simpletransformers
from torch.utils.data import Dataset

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

# ...dataset creation...
# train_dataset = Dataset()
# test_dataset = Dataset()
training_dataset = load_dataset("csv", data_files="training_data.csv")
test_dataset=load_dataset("csv", data_files="testing_data.csv")

training_dataset = training_dataset.remove_columns(["Unnamed: 0", "identifier", "mut", "gene"])
test_dataset = test_dataset.remove_columns(["Unnamed: 0", "identifier", "mut", "gene"])

training_args = TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    fp16=True,
    save_strategy="no",
    weight_decay=0.015,
    output_dir="output"
)


trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
)

_ = trainer.train()