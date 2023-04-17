import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, BertForTokenClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch

def preprocess_function_classification(examples, tokenizer):
    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = examples["label"]
    return inputs

def finetune_event_classification():
    PRETRAINED_MODEL = "bert-base-uncased"
    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL)
    # Load datasets
    event_classification_dataset = load_dataset("csv", data_files={"train": "event_classification.csv"})
    # Preprocess datasets
    event_classification_dataset = event_classification_dataset.map(preprocess_function_classification, fn_kwargs={'tokenizer': tokenizer}, batched=True)
    # Event Classification Model
    sequence_classifier = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)
    # TrainingArguments
    training_args_classification = TrainingArguments(
        "event_classification_checkpoint",
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    # Trainer
    trainer_classification = Trainer(
        sequence_classifier,
        training_args_classification,
        train_dataset=event_classification_dataset["train"],
    )
    # Train models
    trainer_classification.train()
    # Save models
    trainer_classification.save_model("event_classification_model")

# Preprocess function
def preprocess_function(example, tokenizer, label_list):
    text_sequence = example["word"].tolist()
    label_sequence = example["tag"].tolist()
    tokens = tokenizer(text_sequence, is_split_into_words=True, padding=True, truncation=True)
    word_ids = tokens.word_ids()
    
    aligned_labels = [-100 if id is None else label_list.index(label_sequence[id]) for id in word_ids]
    tokens["labels"] = aligned_labels
    return tokens
# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val.iloc[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def ner_finetune():
    # Load data
    #data = load_dataset("csv", data_files={"train": "ner.csv"}).strip().split("\n")

    df = pd.read_csv('ner.csv')

    # Split the data into train and val sets
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # Create a list of unique tags
    label_list = sorted(df["tag"].unique().tolist())

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Convert train and val datasets to the required format
    train_encodings = train_df.groupby("sentence_id").apply(lambda x: preprocess_function(x, tokenizer, label_list))
    val_encodings = val_df.groupby("sentence_id").apply(lambda x: preprocess_function(x, tokenizer, label_list))

    # Trainer
    model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))

    # Training arguments
    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        seed=42,
    )

    # Create dataset
    train_dataset = CustomDataset(train_encodings)
    val_dataset = CustomDataset(val_encodings)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()
    # save model
    trainer.save_model("ner_model")

# main

ner_finetune()
finetune_event_classification()