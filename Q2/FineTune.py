#Created by Lachlan Higgins, c3374994

from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification as SeqClassModel
from transformers import TrainingArguments, Trainer


#model_name = 'bert-base-cased'
#model_name = 'roberta-base'
model_name = 'albert-base-v2'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = load_dataset('cardiffnlp/tweet_topic_single').remove_columns(['date', 'id'])

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


train_dataset = tokenized_datasets['train_coling2022']
eval_dataset = tokenized_datasets['test_coling2022']

model = SeqClassModel.from_pretrained(model_name, num_labels=6).to(device)

#Hyperparameters
training_args = TrainingArguments(output_dir='test_trainer', 
                                  evaluation_strategy='epoch',
                                  weight_decay=0.001,
                                  learning_rate=1e-05,
                                  per_device_eval_batch_size=8,
                                  per_device_train_batch_size=8,
                                  num_train_epochs=4,
                                  warmup_steps=500, #warmup steps for learning scheduler
                                  )


def compute_metrics(p):
    
    labels = p.label_ids
    pred = p.predictions.argmax(-1)
    
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='micro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

