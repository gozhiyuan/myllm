from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate

# Load the dataset
dataset = load_dataset("yelp_review_full")
# print(f"Example from dataset: {dataset['train'][100]}")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Subsample for quick experimentation
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(500))

# Tokenize the subsampled datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)

# Initialize the data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased", num_labels=5, torch_dtype="auto"
)
print(f"Model config: {model.config}")

# Metrics to compute during training
acc_metric = evaluate.load("metric_accuracy.py")
f1_metric = evaluate.load("metric_f1.py")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    acc.update(f1)
    return acc

# Define training arguments
training_args = TrainingArguments(
    output_dir="test_trainer",                          # Output directory for model checkpoints
    per_device_train_batch_size=32,                     # Batch size per device during training
    per_device_eval_batch_size=32,                      # Batch size per device during evaluation
    logging_steps=10,                                   # Number of steps between logging
    evaluation_strategy="epoch",                        # Evaluate at the end of each epoch
    save_strategy="epoch",                              # Save checkpoints at the end of each epoch
    num_train_epochs=4,                                 # Number of training epochs
    save_total_limit=3,                                 # Maximum number of saved checkpoints
    learning_rate=2e-5,                                 # Learning rate
    weight_decay=0.01,                                  # Weight decay for regularization
    metric_for_best_model="f1",                         # Metric to monitor for best model
    load_best_model_at_end=True,                        # Load the best model after training
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
