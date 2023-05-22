from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification as SeqClassModel
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np


dataset = load_dataset('cardiffnlp/tweet_topic_single', split=['train_coling2022','test_coling2022'])

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['test']

#Train

model = SeqClassModel.from_pretrained('roberta-base', num_labels=6)

training_args = TrainingArguments(output_dir='test_trainer', evaluation_strategy='epoch')



metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicitons = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predicitons, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()