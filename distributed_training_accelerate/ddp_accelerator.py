import os
import random
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import Adam
from accelerate import Accelerator
from datasets import load_dataset

# Initialize the Accelerator
accelerator = Accelerator()

# Load the dataset
dataset = load_dataset("yelp_review_full")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased", num_labels=5, torch_dtype="auto"
)

# Define a custom Dataset class
class YelpReviewDataset(Dataset):
    def __init__(self, split):
        self.dataset = dataset[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label']
        return text, label

# Instantiate the datasets
train_dataset = YelpReviewDataset(split='train')
test_dataset = YelpReviewDataset(split='test')

# Function to create a random subset of the dataset
def create_subset_indices(dataset, num_samples):
    indices = list(range(len(dataset)))
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    return indices[:num_samples]

# Create subsets
train_indices = create_subset_indices(train_dataset, 1000)
test_indices = create_subset_indices(test_dataset, 500)

train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

# Define the collate function for tokenization
def collate_fn(batch):
    texts, labels = zip(*batch)
    inputs = tokenizer(
        list(texts),
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    inputs["labels"] = torch.tensor(labels)
    return inputs

# Create DataLoaders
train_loader = DataLoader(
    train_subset,
    batch_size=32,
    collate_fn=collate_fn,
    shuffle=True
)

valid_loader = DataLoader(
    test_subset,
    batch_size=64,
    collate_fn=collate_fn
)

# Optimizer setup
optimizer = Adam(model.parameters(), lr=2e-5)

# Evaluation function
def evaluate(ddp_model, ddp_valid_loader):
    ddp_model.eval()
    acc_num = 0
    # with torch.inference_mode():
    with torch.no_grad():   # torch.inference_mode() will raise error for deepspeed zero3 training. See in the deepspeed notebook
        for batch in ddp_valid_loader:
            output = ddp_model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            # Gather predictions and references from all processes
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            acc_num += (pred.long() == refs.long()).float().sum()
    return acc_num / len(ddp_valid_loader.dataset)

# Training function with local variable renaming
def train(epochs=3, log_step=10):
    global_step = 0
    # Prepare for distributed training, renaming the local variables
    ddp_model, ddp_optimizer, ddp_train_loader, ddp_valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )

    for epoch in range(epochs):
        ddp_model.train()
        for batch in ddp_train_loader:
            ddp_optimizer.zero_grad()
            output = ddp_model(**batch)
            loss = output.loss
            accelerator.backward(loss)
            ddp_optimizer.step()

            if global_step % log_step == 0:
                loss_val = accelerator.reduce(loss, "mean")
                accelerator.print(f"Epoch: {epoch}, global_step: {global_step}, loss: {loss_val.item()}")
            global_step += 1

        acc = evaluate(ddp_model, ddp_valid_loader)
        accelerator.print(f"Epoch: {epoch}, Accuracy: {acc}")

# Start training
train()

# Cleanup
accelerator.end_training()
