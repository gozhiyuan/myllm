import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# Initialize the distributed process group
dist.init_process_group(backend="nccl")  # or "gloo" if you're using CPU

# Load the dataset from Hugging Face
dataset = load_dataset("yelp_review_full")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased", num_labels=5, torch_dtype="auto"
)

# Set device based on LOCAL_RANK environment variable (set by torchrun)
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

# Wrap the model with DistributedDataParallel
model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

# Define a custom Dataset class
class YelpReviewDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label']
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',  # fixed length padding; for dynamic padding, use a collate_fn
            max_length=128,
            return_tensors='pt'
        )
        # Remove the extra batch dimension
        output = {key: val.squeeze(0) for key, val in encoding.items()}
        output["labels"] = torch.tensor(label, dtype=torch.long)
        return output

# Create Dataset instances for train and test splits
train_dataset = YelpReviewDataset(dataset['train'], tokenizer)
test_dataset = YelpReviewDataset(dataset['test'], tokenizer)

# Set a random seed for reproducibility
torch.manual_seed(42)

# Randomly select 100 samples for training and 50 samples for testing
num_train_samples = 1000
num_test_samples = 500

train_indices = torch.randperm(len(train_dataset)).tolist()[:num_train_samples]
test_indices = torch.randperm(len(test_dataset)).tolist()[:num_test_samples]

# Create subsets of the original datasets
train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

# Create DistributedSamplers for the subsets
train_sampler = DistributedSampler(train_subset, shuffle=True)
test_sampler = DistributedSampler(test_subset, shuffle=False)

# Create DataLoaders using the DistributedSamplers
train_loader = DataLoader(train_subset, sampler=train_sampler, batch_size=32)
test_loader = DataLoader(test_subset, sampler=test_sampler, batch_size=32)

# Set up the optimizer
optimizer = Adam(model.parameters(), lr=2e-5)

def train(epochs=3):
    global_step = 0
    for epoch in range(epochs):
        model.train()
        # Set the epoch for the sampler to ensure a different shuffling each epoch
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            # Move all batch tensors to the correct device
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            if global_step % 10 == 0 and dist.get_rank() == 0:
                print(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item()}")
            global_step += 1

    # Save the model checkpoint only on the main process
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), "model_checkpoint.pth")
        print("Model checkpoint saved.")

    # Cleanup distributed processes
    dist.destroy_process_group()

if __name__ == '__main__':
    train()
