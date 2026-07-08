
#multi-head attention

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
    data_files="C:\\Users\\ppriy\\Downloads\\imdbreview\\IMDB Dataset.csv"
)

def filter_empty(example):
    return example["review"].strip() != ""

dataset = dataset.filter(filter_empty)

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
    remove_columns=["review"]
)

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"]
)

dataloader = DataLoader(tokenized_dataset['train'], batch_size=64, shuffle=False)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # same projections, but for full d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, X, mask=None):
        batch_size, seq_len, _ = X.shape

        # 1. Linear projections
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        # 2. Split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 3. Transpose to (batch, heads, seq, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 4. Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        import math
        scores = scores / math.sqrt(self.head_dim)

        weights = torch.softmax(scores, dim=-1)

        out = torch.matmul(weights, V)

        # 5. Concatenate heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.d_model)

        # 6. Final linear layer
        out = self.fc_out(out)

        return out, weights
    
#transformer block 

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, X, attention_mask=None):
        # Multi-head attention
        attn_out, weights = self.attention(X, attention_mask)

        # Residual + LayerNorm
        X = self.norm1(X + attn_out)

        return X, weights

batch = next(iter(dataloader))

input_ids = batch["input_ids"]
attention_mask = batch["attention_mask"]

vocab_size = tokenizer.vocab_size
d_model = 64

embedding = nn.Embedding(vocab_size, d_model)
X = embedding(input_ids)

block = TransformerBlock(d_model, num_heads=4)
output, weights = block(X, attention_mask)

print("Input shape:", X.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)

attn = weights[0][:10, :10].detach().cpu().numpy()
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

filtered = [(i, t) for i, t in enumerate(tokens) if t != "<|endoftext|>"]

indices = [i for i, _ in filtered][:10]
tokens = [t for _, t in filtered][:10]

tokens = [t.replace("Ġ", "") for t in tokens]
tokens = [t.replace("Ċ", "\\n") for t in tokens]

head = 0  # choose which head to visualize
attn = weights[0][head][indices][:, indices].detach().cpu().numpy()

# plot
plt.figure()
plt.imshow(attn)
plt.colorbar()

plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.yticks(range(len(tokens)), tokens)

plt.title("Multi-Head Attention Heatmap")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")

plt.show()
weights[0][0]   # head 0
weights[0][1]   # head 1

for h in range(4):
    attn = weights[0][h][indices][:, indices].detach().cpu().numpy()

    plt.figure()
    plt.imshow(attn)
    plt.title(f"Head {h}")
    plt.colorbar()
    plt.show()

