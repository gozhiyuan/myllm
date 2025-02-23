import os
import random
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import Adam
from accelerate import Accelerator
from datasets import load_dataset

# Initialize the Accelerator
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=2, 
    log_with="tensorboard", project_dir="logs")
accelerator.init_trackers("runs")

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

# # Function to ensure only rank 0 prints log messages
# # Can be replace by accelerator.print
# def print_rank_0(info):
#     if accelerator.is_local_main_process:
#         print(info)

# Evaluation function
def evaluate(ddp_model, ddp_valid_loader):
    ddp_model.eval()
    acc_num = 0
    with torch.no_grad():   # torch.inference_mode() will raise error for deepspeed zero3 training
        for batch in ddp_valid_loader:
            output = ddp_model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            # Gather predictions and references
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            # Ensure predictions and references are on the same device for comparison
            acc_num += (pred.long() == refs.long()).float().sum()

    return acc_num / len(ddp_valid_loader.dataset)


# Training function
def train(epochs=3, log_step=100, resume=None):
    global_step = 0
    ddp_model, ddp_optimizer, ddp_train_loader, ddp_valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )

    resume_step = 0
    resume_epoch = 0

    if resume is not None:
        accelerator.load_state(resume)
        steps_per_epoch = math.ceil(len(trainloader) / accelerator.gradient_accumulation_steps)
        resume_step = global_step = int(resume.split("step_")[-1])
        resume_epoch = resume_step // steps_per_epoch
        resume_step -= resume_epoch * steps_per_epoch
        accelerator.print(f"resume from checkpoint -> {resume}")

    for epoch in range(epochs):
        ddp_model.train()
        if resume and ep == resume_epoch and resume_step != 0:
            active_dataloader = accelerator.skip_first_batches(ddp_train_loader, resume_step * accelerator.gradient_accumulation_steps)
        else:
            active_dataloader = ddp_train_loader
        for batch in active_dataloader:
            with accelerator.accumulate(ddp_model):
                ddp_optimizer.zero_grad()
                output = ddp_model(**batch)
                loss = output.loss
                accelerator.backward(loss)  # accelerator backward
                ddp_optimizer.step()

                if accelerator.sync_gradients:
                    global_step += 1

                    if global_step % log_step == 0:
                        loss = accelerator.reduce(loss, "mean")
                        accelerator.print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
                        accelerator.log({"loss": loss.item()}, global_step)

                    if global_step % 10 == 0 and global_step != 0:
                        accelerator.print(f"save checkpoint -> step_{global_step}")
                        accelerator.save_state(accelerator.project_dir + f"/step_{global_step}")
                        accelerator.unwrap_model(ddp_model).save_pretrained(
                            save_directory=accelerator.project_dir + f"/step_{global_step}/model",
                            is_main_process=accelerator.is_main_process,
                            state_dict=accelerator.get_state_dict(ddp_model),
                            save_func=accelerator.save
                        )

        # Evaluate the model after each epoch
        acc = evaluate(ddp_model, ddp_valid_loader)
        accelerator.print(f"Epoch: {epoch}, Accuracy: {acc}")

# Start training
train()

# Cleanup
accelerator.end_training()
