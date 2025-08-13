from datasets import load_dataset

dataset = load_dataset('imdb')

# print(dataset)
# print(dataset['train'][101])

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding = 'max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(tokenized_datasets['train'][101]['input_ids'])

small_train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(200))
small_eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(200))

import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

import numpy as np
import evaluate

metric = evaluate.load('accuracy')

def compute_metric(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir='test_trainer', num_train_epochs= 2, evaluation_strategy='epoch')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metric
)

trainer.train()

trainer.save_model('model/sentiment-classifier')

from huggingface_hub import notebook_login
notebook_login()

trainer.push_to_hub('valen/sentiment-classifier')

from transformers import BertConfig, BertModel

model = AutoModelForSequenceClassification.from_pretrained('models/sentiment-classifier')
inputs = tokenizer()

outputs = model(**inputs)

from transformers import pipeline

classifier = pipeline(task='sentiment-analysis', model=model, tokenizer=tokenizer)
classifier('I cannot believe you did it again!')