"""Download huggingface data and models to cache local"""

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5, torch_dtype="auto")
