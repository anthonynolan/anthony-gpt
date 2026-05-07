import torch.nn as nn
import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import string
# import numpy as np
# import pandas as pd
# import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
# import re
import tiktoken
from torch.utils.data import TensorDataset, DataLoader
# from pprint import pprint
from tqdm import tqdm
import time
from pathlib import Path

torch.random.manual_seed(1234)


def save_model(model_dir, model, file_name="model.torch"):
    with open(Path(model_dir, file_name), 'wb') as f:
        torch.save(model, f)

def load_model(model_dir, file_name="model.torch"):
    with open(Path(model_dir, file_name), 'rb') as f:
        model = torch.load(f, weights_only=False)
    return model

def tokenise_text(text):
    tokens = torch.tensor(enc.encode(text), device=device)
    return tokens

def load_data():

    with open('complete-jane-austen.txt') as f:
        content = f.read()
        content = ' '.join(content.split())
        print(f'content length in chars: {len(content):,}')
        print(f'sample of content: {content[:100]}')
        encoded_text = tokenise_text(content)
        print(f'sample of encoded text: {encoded_text[:100]}')
        print(f'number of tokens: {len(encoded_text):,}')

    xs = []
    ys = []
    for i in range(0, len(encoded_text)-context_length):
        x_chunk = encoded_text[i:i+context_length]
        y_chunk = encoded_text[i+1:i+context_length+1]

        xs.append(x_chunk)
        ys.append(y_chunk)

    X = torch.stack(xs)
    Y = torch.stack(ys)
    split_index = int(X.shape[0]*0.999)
    X_train = X[:split_index]
    Y_train = Y[:split_index]
    X_val = X[split_index:]
    Y_val = Y[split_index:]
    
    dataset_train = TensorDataset(X_train, Y_train)
    dataset_val = TensorDataset(X_val, Y_val)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    
    return loader_train, loader_val


def generate(model, max_words, prompt_text):
    
    idx = tokenise_text(prompt_text).unsqueeze(0)
    if idx.shape[0]< context_length:
        prompt_text+=(' '*(context_length-idx.shape[0]))
        idx = tokenise_text(prompt_text).unsqueeze(0)
    
    for _ in range(max_words):
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_idx), dim=1)

        output = enc.decode(idx[0].tolist()) 
    return output

class Head(nn.Module):
    def __init__(self, head_dimension, d_model):
        super().__init__()

        self.query = nn.Linear(d_model, head_dimension, bias=False)
        self.key = nn.Linear(d_model, head_dimension, bias=False)
        self.value = nn.Linear(d_model, head_dimension, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(0.1)
        self.tril: torch.Tensor
        

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1)*(C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out =  wei @ v

        return out
    
    
class TransformerNameGenerator(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, n_heads):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)

        head_dim = d_model//n_heads
        self.heads = nn.ModuleList([Head(head_dim, d_model) for _ in range(n_heads)])
        self.proj = nn.Linear(d_model, d_model)

        self.llm_model = nn.Linear(d_model, vocab_size)
        
        # Tie the output embedding vectors to those learned at the input
        # increases efficiency. They are from the same space.
        if tie_weights:
            self.llm_model.weight = self.token_embedding.weight

        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))
            

    def forward(self, idx):
        B, T = idx.shape
 
        token_embeddings = self.token_embedding(idx) # B, T, d_model
 
        pos = torch.arange(T, device=device)
 
        positional_embeddings = self.position_embedding(pos) # T, d_model
 
        x = token_embeddings + positional_embeddings

        output = torch.concat([head(x) for head in self.heads], dim=-1)

        attn_output = self.proj(output)

        x = x + attn_output

        x = x + self.ffn(self.ln_3(x))
        
        logits = self.llm_model(x)

        return logits


if __name__=='__main__':
    enc = tiktoken.get_encoding('gpt2')
    vocab_size = enc.n_vocab
    print(f'vocab_size: {vocab_size:,}')

    context_length = 64
    d_model = 64
    n_heads = 16
    batch_size=32

    model_dir = "./models"

    # experiment parameters
    tie_weights = False

    model_file_name = "multihead.torch"


    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f'device: {device}')


    model = TransformerNameGenerator(vocab_size, context_length, d_model, n_heads=n_heads)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)

    loader_train, loader_val = load_data()

    writer = SummaryWriter()

    pbar = tqdm(loader_train)
    epochs = 1

    model.train()
    tokens_per_second = 0
    for epoch in range(epochs):
        for i, (xb, yb) in enumerate(pbar):
            torch.cuda.synchronize()
            start = time.time()
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model.forward(xb)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = yb.view(B*T)
            loss = F.cross_entropy(logits, targets)
            writer.add_scalar("Loss/train", loss, i)
            if not i%500:
                # print(loss.item(), f'Epoch: {epoch}, Batch: {i}')
                with torch.no_grad():
                    model.eval()
                    val_loss=0
                    for xb, yb in loader_val:
                        if xb.shape[0]==batch_size:
                            val_loss += F.cross_entropy(model.forward(xb).view(B*T, C), yb.view(B*T)) 
                    pbar.set_postfix(val_loss=(val_loss/len(loader_val)).item(), loss=loss.item(), tokens_per_second=tokens_per_second)
                    writer.add_scalar("Loss/val", val_loss/len(loader_val), i)

                    output = generate(model, 50, "")
                    print(f'in training generation: {output}')

                    model.train()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            step_time = time.time() - start
            tokens_per_second = batch_size*context_length/step_time
            # print(tokens_per_second)

    save_model(model_dir, model, file_name=model_file_name)
