# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: news_classifier
#     language: python
#     name: python3
# ---

# %% [markdown]
# # News Classifier with StreamLit App
# Build a news classifier model using news article from Huffing Post from <a href="https://www.kaggle.com/datasets/rmisra/news-category-dataset">Kaggle News Category Dataset</a> across 42 categories. </br>
# We will fine-tune modern bert model with a context length of 8192

# %%
import lzma
import pickle
import hashlib
import os
import pandas as pd
import numpy as np
import requests as r
import seaborn as sns
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from joblib import hash
import torch

# Configure matplotlib
mpl.rcParams['figure.figsize'] = (12, 6)
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['xtick.labelsize'] = 'medium'
mpl.rcParams['ytick.labelsize'] = 'medium'
warnings.simplefilter('ignore')
mpl.style.use('ggplot')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# %% [markdown]
# ## Source Data

# %%
import zipfile
import json
from io import TextIOWrapper

def load_json_from_zip(zip_path: str, zip_filename: str) -> pd.DataFrame:
    """
    Opens a zip file, reads a JSON file inside, and loads it into a pandas DataFrame.
    
    Parameters:
        zip_path (str): Path to the .zip file.

    Returns:
        pd.DataFrame: DataFrame created from the JSON file.
    """
    path_to_zip_file = zip_path + zip_filename
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(zip_path) 
        
        # Find all files with .json extension
        json_filenames = [name for name in zip_ref.namelist() if name.endswith('.json')]
        json_filename = zip_path+json_filenames[0]  # assumes only one JSON file
        df = pd.read_json(json_filename, lines=True, encoding='utf-8')
        os.system(f'rm {json_filename}')  # remove the extracted JSON file
    
    return df

#Example usage:
df = load_json_from_zip(zip_path="../data/",zip_filename="news-category-dataset.zip")
df.head()

# %% [markdown]
# ## Exploratory Analysis

# %%
# Exploratory analysis of the dataset
df.category.value_counts().plot(kind='bar', figsize=(12, 6))
plt.title('Distribution of News Categories')
plt.xlabel('Category')
plt.ylabel('Number of Articles')
plt.xticks(rotation=90)
plt.show()

df.columns
df["text"] = df["headline"] + "\n" + df["short_description"]
df.head()

# %%
# Convert the dataset into a huggingface dataset
from datasets import Dataset, ClassLabel
dataset = Dataset.from_pandas(df[['text', 'category']])

# Obtain label as ClassLabel column from category column in dataset and add it to the dataset
# This will convert the 'category' column into a ClassLabel type, which is useful for classification tasks. 
label_list = dataset.unique('category')
num_labels = len(label_list)
class_label = ClassLabel(num_classes=num_labels, names=label_list)
dataset = dataset.cast_column('category', class_label)
dataset = dataset.rename_column('category', 'label')  # Rename column for consistency

# Split the dataset into train and test sets
train_test_split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Split the train dataset into train and validation sets
train_val_split = train_dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

# Print the number of samples in each dataset
print(f"Train dataset size: {len(train_dataset)}") 
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# %%
from transformers import AutoTokenizer

max_length = 256  # Maximum length for tokenization, can be adjusted based on model requirements

# Model id to load the tokenizer
model_id = "answerdotai/ModernBERT-base"  # You can change this to any other model id

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
 
# Tokenize helper function
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt", max_length=max_length)
 
# Tokenize dataset
tokenized_dataset = train_val_split.map(tokenize, batched=True, remove_columns=["text"], batch_size=64) # type: ignore
tokenized_dataset["train"].features.keys()

# Test tokenize dataset
test_tokenized_dataset = test_dataset.map(tokenize, batched=True, remove_columns=["text"], batch_size=64) # type: ignore
test_tokenized_dataset.features.keys()

# %% [markdown]
# ## Prepare label 2 id and id 2 label and load model

# %%
from transformers import AutoModelForSequenceClassification 
 
# Prepare model labels - useful for inference
labels = tokenized_dataset["train"].features["label"].names
print(labels)
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# Download the model from huggingface.co/models
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=num_labels, label2id=label2id, id2label=id2label, attn_implementation="eager", reference_compile=False)

# %% [markdown]
# ## Evaluate Model for F1 score

# %%
import numpy as np
from sklearn.metrics import f1_score
 
# Metric helper method
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
            labels, predictions, labels=labels, pos_label=1, average="weighted"
        )
    return {"f1": float(score) if score == 1 else score}


# %% [markdown]
# ## Fine-tune the classifier

# %%
from huggingface_hub import HfFolder
from transformers import Trainer, TrainingArguments
 
# Define training args
training_args = TrainingArguments(
    output_dir= "../models/modernbert-news-classifier",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    learning_rate=1e-4,
    num_train_epochs=10,
    bf16=True, # bfloat16 training
    optim="adamw_torch_fused", # improved optimizer 
    # logging & evaluation strategies
    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1, # keep only the latest checkpoint
    load_best_model_at_end=True,
    # evaluation metrics
    metric_for_best_model="f1",
    # push to hub parameters
    push_to_hub=False
)

## Create a Trainer instance
#trainer = Trainer(
#    model=model,
#    args=training_args,
#    train_dataset=tokenized_dataset["train"],
#    eval_dataset=tokenized_dataset["test"],
#    compute_metrics=compute_metrics,
#)
#trainer.train()

# %%

# %%
# Load the saved huggingface model and test on the test dataset
from transformers import pipeline

# Load the model from the saved directory
model = AutoModelForSequenceClassification.from_pretrained("../models/modernbert-news-classifier", local_files_only=True)

# Create a pipeline for text classification
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device='cuda:0')

# Test the classifier on the test dataset
test_predictions = classifier(test_dataset["text"], truncation=True, padding=True, max_length=max_length)
# Print the first 5 predictions
for i in range(5):
    print(f"Text: {test_dataset['text'][i]}")
    print(f"Predicted label: {test_predictions[i]['label']}, Score: {test_predictions[i]['score']:.4f}\n") # type: ignore


# %%
# Calculate the weighted F1 score on the test predictions
from sklearn.metrics import f1_score

test_labels = test_dataset["label"]
f1 = f1_score(test_labels, [int(label2id[pred['label']]) for pred in test_predictions], average='weighted') # type: ignore
print(f"Weighted F1 Score on Test Set: {f1:.4f}")

