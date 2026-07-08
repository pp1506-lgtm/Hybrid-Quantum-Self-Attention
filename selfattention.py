#self attention 

import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from datasets import load_dataset

dataset = load_dataset(
    "csv",
    data_files="C:\\Users\\ppriy\\Downloads\\imdbreview\\IMDB Dataset.csv",
    escapechar="\\",
    on_bad_lines="skip"
)

def filter_empty(example):
    return example["review"] is not None and example["review"].strip() != ""

dataset["train"] = dataset["train"].filter(filter_empty)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["review",'sentiment']
)

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"]
)

dataloader = DataLoader(tokenized_dataset["train"], batch_size=64, shuffle=True)

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, X, mask=None):
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        scores = torch.matmul(Q, K.transpose(-2, -1))

        d_k = Q.size(-1)
        scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        return output, weights

batch = next(iter(dataloader))

input_ids = batch["input_ids"]
atten_mask = batch["attention_mask"]

vocab_size = tokenizer.vocab_size
d_model = 64

embedding = nn.Embedding(vocab_size, d_model)
X = embedding(input_ids)

atten = SelfAttention(d_model)
output, weights = atten(X, atten_mask)

print("Input shape:", X.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)

attn = weights[0][:10, :10].detach().cpu().numpy()
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

filtered = [(i, t) for i, t in enumerate(tokens) if t != "<|endoftext|>" ]

indices = [i for i, _ in filtered][:10]
tokens = [t for _, t in filtered][:10]

tokens = [t.replace("Ġ", "") for t in tokens]
tokens = [t.replace("Ċ", "\\n") for t in tokens]

attn = weights[0][indices][:, indices].detach().cpu().numpy()

# plot
plt.figure()
plt.imshow(attn)
plt.colorbar()

plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.yticks(range(len(tokens)), tokens)

plt.title("Self-Attention Heatmap")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")

plt.show()
print(weights[0][0])