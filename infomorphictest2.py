import os
import torch
import torch.nn as nn
import torch.fft
import logging
import math
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import sys
from transformers import PreTrainedTokenizerFast
import re
import torch.utils.checkpoint as checkpoint
import random
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# Use this queue with custom SYCL ops or wrappers

# Check device properties
print(device)

seq_len = 100

########################################
# 1. Build a Byte-Level Tokenizer/Vocab
########################################

from transformers import PreTrainedTokenizerFast

# 🔹 Change this to the actual path where your BPE tokenizer files are stored
tokenizer_path = r"C:\Users\Austin\.cursor\ruletransformer-main\mhlatest-main"  

# 🔹 Load a BPE tokenizer from local files
base_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

print(f"✅ Loaded custom BPE tokenizer from: {tokenizer_path}")
print(f"📏 Vocabulary size: {base_tokenizer.vocab_size}")

# Wrap it with the hierarchical tokenizer
tokenizer = base_tokenizer


########################################
# 2. Data Extraction
########################################

def extract_data(json_data):
    """Extracts training data from JSON file and tokenizes it."""
    input_ids_list = []
    target_ids_list = []

    for item in json_data:
        conversations = item.get("conversations", [])

        if not isinstance(conversations, list) or len(conversations) < 2:
            print(f"⚠️ Skipping entry with no valid conversation: {item}")
            continue

        for i in range(len(conversations) - 1):
            user_turn = conversations[i]
            assistant_turn = conversations[i + 1]

            # Ensure we only process valid user-assistant exchanges
            if user_turn.get("from") in ["user", "human"] and assistant_turn.get("from") in ["assistant", "gpt"]:
                query = user_turn.get("value", "").strip()
                target = assistant_turn.get("value", "").strip()

                # 🔹 Ensure valid text exists before tokenizing
                if not query or not target:
                    print(f"⚠️ Skipping empty user/assistant exchange: {user_turn} -> {assistant_turn}")
                    continue  

                input_ids = tokenizer.tokenize(query)
                target_ids = tokenizer.tokenize(target)

                # 🔹 Ensure tokenized output isn't empty
                if not input_ids or not target_ids:
                    print(f"⚠️ Skipping invalid tokenized entry: {query} -> {input_ids}")
                    continue

                input_ids_list.append(input_ids)
                target_ids_list.append(target_ids)
    

    return list(zip(input_ids_list, target_ids_list))  # Ensure format is (input, target)

def load_dataset(dataset_path):

            dataset_files = os.listdir(dataset_path)
            query_target_pairs = []

            for file in dataset_files:
                file_path = os.path.join(dataset_path, file)
                if file.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        text_data = list
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df.strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                elif file.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.jsonl'):
                                for line in f:
                                    conversation = json.loads(line.strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                            else:
                                data = json.load(f)
                                query_target_pairs.extend(extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]

                elif file.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['text'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'TEXT' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['TEXT'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'messages' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['messages'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                elif file.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        text_data.append(text)
                else:
                    print("errpr")
            if not query_target_pairs:
                print("Error", "No valid query/target pairs found in the dataset.")
                return

            # Store text data for saving as a text file
            text_data = []
            for query, target in query_target_pairs:
                text_data.append(f"User: {query}\nAssistant: {target}")

            logging.info(f"Loaded dataset with {len(query_target_pairs)} query/target pairs.")
            return query_target_pairs


def extract_query_target_pairs( data):
        query_target_pairs = []

        for conversation in data:
            if conversation.get("messages"):
                messages = conversation.get("messages", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
                        query = messages[i].get("content") or messages[i].get("value", "")
                        target = messages[i + 1].get("content") or messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))

                    elif messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))

            elif conversation.get("conversations"):
                messages = conversation.get("conversations", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            elif conversation.get("text"):
                messages = conversation.get("text", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            else:
                user_messages = conversation.get("user", [])
                assistant_messages = conversation.get("assistant", [])
                for i in range(min(len(user_messages), len(assistant_messages))):
                    query = user_messages[i].replace('\n', ' ').strip()
                    target = assistant_messages[i].replace('\n', ' ').strip()
                    query_target_pairs.append((query, target))
            # Final fallback: split everything into sequence-length chunks for predictive text
            if not query_target_pairs:
                all_text = " ".join([m.get("text", "") for conversation in data for m in conversation])
                tokenized_text = tokenizer.encode(all_text, truncation=False)
                query_target_pairs = [
                    {"query": tokenized_text[i:i+seq_len], "target": tokenized_text[i:i+seq_len]}
                    for i in range(0, len(tokenized_text), seq_len)
                ]

        return query_target_pairs

def tokenize_data(query_target_pairs):

        # Select training mode
        input_ids_list = []  # Initialize for unchunked dataset
        labels_list = []  # Initialize for unchunked dataset

        for query, target in query_target_pairs:
                        input_ids, labels = _generate_training_pairs(query, target)

                        if input_ids and labels:
                            input_ids_list.append(input_ids)  # Store for training
                            labels_list.append(labels)  # Store for training
                            #print (input_ids)
                            #print(labels)
        return input_ids_list, labels_list


def _generate_training_pairs(query, target):
        # Debugging logs
        logging.debug(f"Generating Training Pairs - Query: {query}")
        logging.debug(f"Generating Training Pairs - Target: {target}")

        # Ensure inputs are valid strings before tokenization
        query_ids = tokenizer.encode(str(query) if query else "", truncation=True, max_length=seq_len)
        target_ids = tokenizer.encode(str(target) if target else "", truncation=True, max_length=seq_len)

        input_ids = [tokenizer.bos_token_id] + query_ids + [tokenizer.eos_token_id]
        labels = [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]

        return input_ids, labels

def prepare_batch(input_ids, labels, seq_len):
                pad_token_id = tokenizer.pad_token_id if tokenizer else pad_token_id  # Default to global if tokenizer isn't set      
                max_length = seq_len  # Adjust as needed
                logging.info("max_length set")
                # Convert lists of token IDs to tensors and calculate original sequence lengths

                #input_ids = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in input_ids]
                #labels = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in labels]

                # ✅ Compute correct padding lengths
                #input_ids = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in input_ids]
                #labels = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in labels]
                
                input_ids = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length]
                    for tokens in input_ids
                ]
                logging.info("input ids torched to tensor")
                print(input_ids)
                labels = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length]
                    for tokens in labels
                ]
                logging.info("labels torched to tensor")
                print(labels)
                # Stack tensors
                input_ids = torch.stack(input_ids).to(device)
                labels = torch.stack(labels).to(device)
                data = torch.utils.data.TensorDataset(input_ids, labels)
                return data


########################################
# 3. Dataset and Collate Function
########################################

class ChatDataset(Dataset):
    def __init__(self, json_data, tokenizer, max_seq_length):
        """Initialize dataset and tokenize the data properly."""
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # 🔹 Ensure data is correctly processed
        self.data = extract_data(json_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Returns exactly two elements: (input, target)."""
        return self.data[idx]

def collate_fn2(batch, max_length, tokenizer):
    src_batch, tgt_batch = zip(*batch)

    pad_token_id = tokenizer.pad_token_id or 0  # Ensure pad token is valid

    src_batch = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in src_batch]
    tgt_batch = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in tgt_batch]

    # ✅ Compute correct padding lengths
    src_batch = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in src_batch]
    tgt_batch = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in tgt_batch]
    print(src_batch)
    print(tgt_batch)
    return torch.stack(src_batch), torch.stack(tgt_batch)

def collate_fn(batch):
    """
    Collate function for standard seq2seq data. Each sample is a tuple (input_ids, target).
    Both sequences are padded/truncated to a fixed length.
    """
    input_ids = []
    labels = []
    seq_lengths = []

    if len(batch[0]) == 2:
        for query, target in batch:
            input_ids.append(query)
            labels.append(target)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        return input_ids, labels
    if len(batch[0]) == 3:
        # Dataset returns: input_ids, labels, seq_lengths
        input_ids, labels, seq_lengths = zip(*batch)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        return input_ids, labels, seq_lengths

##############################################
# Positional Encoding (Standard Sin/Cos Version)
##############################################



# === Hypertransformer ===
class HyperTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, num_heads, weight_dim):
        super().__init__()
        self.embed = nn.Linear(input_dim, model_dim)
        encoder = nn.TransformerEncoderLayer(model_dim, num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.out = nn.Linear(model_dim, weight_dim)

    def forward(self, x):
        x.to(device)
        x = self.embed(x.float().to(device)).to(device)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.out(pooled)

class DualWeightLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_fwd = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.W_bwd = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, direction="forward"):
        if direction == "forward":
            return F.linear(x, self.W_fwd, self.bias)
        elif direction == "backward":
            return F.linear(x, self.W_bwd.t())  # interpret as error or influence backward
        else:
            raise ValueError("direction must be 'forward' or 'backward'")

class DualWeightMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = DualWeightLayer(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.layer2 = DualWeightLayer(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(self.act(self.layer1(x, direction="forward")), direction="forward")

    def backward_pass(self, loss_grad):
        """
        Simulated backward dynamics using backward weights.
        Could be an activation map or error signal flowing in reverse.
        """
        x = self.layer2(loss_grad, direction="backward")
        x = self.act(x)
        x = self.layer1(x, direction="backward")
        return x
    
class DualWeightLayerWithPlasticity(DualWeightLayer):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        self.plasticity = nn.Parameter(torch.zeros(out_dim, in_dim))

    def forward(self, x, direction="forward", apply_plasticity=False):
        W = self.W_fwd
        if apply_plasticity:
            W = W + self.plasticity * torch.bmm(x.unsqueeze(2), x.unsqueeze(1)).mean(0)
        return F.linear(x, W, self.bias)


class DualAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Forward attention
        self.qkv_proj_fwd = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_out_proj_fwd = nn.Linear(embed_dim, embed_dim)

        # Backward attention
        self.qkv_proj_bwd = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_out_proj_bwd = nn.Linear(embed_dim, embed_dim)

        # Shared normalization and FFN
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def attention(self, x, qkv_proj, out_proj):
        B, T, C = x.size()
        qkv = qkv_proj(x)  # shape: (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for multi-head attention
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return out_proj(attn_output)

    def forward(self, x, direction="forward"):
        residual = x

        if direction == "forward":
            x = self.attention(x, self.qkv_proj_fwd, self.attn_out_proj_fwd)
        elif direction == "backward":
            x = self.attention(x, self.qkv_proj_bwd, self.attn_out_proj_bwd)
        else:
            raise ValueError("direction must be 'forward' or 'backward'")

        x = self.norm1(x + residual)
        x = self.norm2(x + self.ff(x))
        return x

class DualTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, num_layers=2, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            DualAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.out = nn.Linear(embed_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, x, direction="forward"):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, direction=direction)
        return self.out(x)


class PlasticDualAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, plasticity=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.plasticity_enabled = plasticity

        # Attention projections
        self.qkv_proj_fwd = nn.Linear(embed_dim, 3 * embed_dim)
        self.qkv_proj_bwd = nn.Linear(embed_dim, 3 * embed_dim)

        self.out_proj_fwd = nn.Linear(embed_dim, embed_dim)
        self.out_proj_bwd = nn.Linear(embed_dim, embed_dim)

        # Plasticity gate (learnable gating function)
        self.plasticity_gate_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def multihead_attention(self, x, qkv_proj, out_proj):
        B, T, C = x.shape
        H = self.num_heads
        D = C // H

        qkv = qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / D**0.5
        attn_weights = F.softmax(attn, dim=-1)
        attn_out = attn_weights @ v  # (B, H, T, D)

        out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return out_proj(out), attn_out  # return attention output for update use

    def forward(self, x, direction="forward"):
        residual = x
        B, T, C = x.shape

        if direction == "forward":
            attn_out, fwd_heads = self.multihead_attention(x, self.qkv_proj_fwd, self.out_proj_fwd)
            x = self.norm1(attn_out + residual)
            x = self.norm2(self.ff(x) + x)
            self.last_fwd_heads = fwd_heads.detach()

            return x  # standard output

        elif direction == "backward" and self.plasticity_enabled:
            attn_out, bwd_heads = self.multihead_attention(x, self.qkv_proj_bwd, self.out_proj_bwd)

            # compute plasticity gate signal
            gate_input = x.mean(dim=1)  # (B, C)
            gate = torch.sigmoid(self.plasticity_gate_proj(gate_input))  # (B, 1)
            print(f"Plasticity Gate (mean across batch): {gate.mean().item():.4f}")

            # Hebbian-like update: outer product between attention heads from fwd and bwd
            # Simulating: ΔW = η * x_fwdᵀ · x_bwd
            if hasattr(self, "last_fwd_heads"):
                #shape for heads is batch-size, embed/n_heads, seq_len, embed/n_heads
                hebb_update = torch.einsum('bhte,bhue->bthe', self.last_fwd_heads, bwd_heads)
                print (hebb_update.shape)
                b, s, d, d = hebb_update.size()
                hebb_update = hebb_update.reshape(b, s, d*d) # (b, s, e/h, e/h) -> (b,s,e)
                # Project to match forward weight shape
                print(hebb_update.shape)
                update_tensor = hebb_update.mean(dim=(0,1))  # crude average per-head, shape (b, s, d)
                print(update_tensor.shape)
                #print(self.qkv_proj_fwd.weight.shape) (d, 3*d)
                #update_tensor = update_tensor.view_as(self.qkv_proj_fwd.weight[:self.embed_dim])  # align with W_q
                self.qkv_proj_fwd.weight.data[:self.embed_dim] += 0.001 * gate.mean() * update_tensor

            self.last_bwd_heads = bwd_heads.detach()
            return x  # backward signal output

        elif direction == "backward":
            # If not updating, just return backward attention output
            attn_out, _ = self.multihead_attention(x, self.qkv_proj_bwd, self.out_proj_bwd)
            return attn_out

        else:
            raise ValueError("Unknown direction.")

class PlasticDualTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, num_layers=2, num_heads=4, plasticity=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            PlasticDualAttentionBlock(embed_dim, num_heads, plasticity=plasticity)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(embed_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, x, direction="forward"):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, direction=direction)
        return self.out(x)



class InfomorphicNeuron(torch.nn.Module):
    def __init__(self, input_dim_R, input_dim_C):
        super().__init__()
        self.wR = torch.nn.Parameter(torch.randn(input_dim_R) * 0.01)
        self.wC = torch.nn.Parameter(torch.randn(input_dim_C) * 0.01)
        self.bias_R = torch.nn.Parameter(torch.zeros(1))
        self.bias_C = torch.nn.Parameter(torch.zeros(1))

    def forward(self, xR, xC):
        r = F.linear(xR, self.wR, self.bias_R)
        c = F.linear(xC, self.wC, self.bias_C)

        # Example sigmoid-based modulated activation
        A = r * (0.5 + torch.sigmoid(2 * r * c))
        prob = torch.sigmoid(A)
        return prob  # returns probability of firing HIGH (+1)

    def compute_pid_objective(self, xR, xC, y_true, gamma):
        """
        Placeholder for PID-based gradient estimate.
        """
        # This requires estimating the 5 PID atoms over a minibatch
        Iunq_R, Iunq_C, Ired, Isyn, Hres = estimate_pid_atoms(xR, xC, y_true)

        # Weighted local goal
        G = (gamma['unq_R'] * Iunq_R +
             gamma['unq_C'] * Iunq_C +
             gamma['red'] * Ired +
             gamma['syn'] * Isyn +
             gamma['res'] * Hres)
        return G

def estimate_pid_atoms(xR, xC, y):
    # This is a placeholder — real PID needs histograms or kernel density estimation
    # Use Makkeh et al.'s `I_sx∩` estimator in practical implementation
    return torch.tensor(0.1), torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.1), torch.tensor(0.05)

def example_neruon_and_batch():
    # Example neuron and batch
    neuron = InfomorphicNeuron(input_dim_R=784, input_dim_C=1)
    xR = torch.rand(32, 784)  # batch of 32 MNIST images
    xC = torch.ones(32, 1) * 1  # context: label bit (1 or -1)
    y_true = torch.randint(0, 2, (32,)) * 2 - 1  # target output {-1, 1}

    # Define goal weighting
    gamma = {
        'unq_R': 0.1,
        'unq_C': 0.1,
        'red': 1.0,
        'syn': 0.1,
        'res': 0.0
    }

    # Forward pass
    probs = neuron(xR, xC)
    loss = neuron.compute_pid_objective(xR, xC, y_true, gamma)
    loss.backward()


def bin_inputs(X, n_bins=5):
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    return est.fit_transform(X.detach().cpu().numpy())

def empirical_joint_probs(x_r, x_c, y):
    data = np.stack([x_r, x_c, y], axis=1)
    unique, counts = np.unique(data, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return unique, probs

def estimate_mutual_information(x, y):
    return mutual_info_score(x, y)

def estimate_pid_atoms(xR, xC, Y, bins=5):
    # Bin all variables
    xR_binned = bin_inputs(xR, bins)[:, 0]
    xC_binned = bin_inputs(xC, bins)[:, 0]
    y_binned = (Y.detach().cpu().numpy() > 0.5).astype(int)

    # Mutual Information Estimates
    I_Y_XR = estimate_mutual_information(y_binned, xR_binned)
    I_Y_XC = estimate_mutual_information(y_binned, xC_binned)
    I_Y_XR_XC = estimate_mutual_information(y_binned, xR_binned + xC_binned * bins)

    # Approximate PID atoms (heuristic decomposition)
    I_red = max(0, min(I_Y_XR, I_Y_XC))
    I_unq_R = max(0, I_Y_XR - I_red)
    I_unq_C = max(0, I_Y_XC - I_red)
    I_syn = max(0, I_Y_XR_XC - I_red - I_unq_R - I_unq_C)
    Hres = 1.0 - I_Y_XR_XC  # entropy of Y minus info explained

    return (
        torch.tensor(I_unq_R, dtype=torch.float32),
        torch.tensor(I_unq_C, dtype=torch.float32),
        torch.tensor(I_red, dtype=torch.float32),
        torch.tensor(I_syn, dtype=torch.float32),
        torch.tensor(Hres, dtype=torch.float32),
    )

# ---------------- Infomorphic Neuron ----------------

class InfomorphicNeuron(nn.Module):
    def __init__(self, input_dim_R, input_dim_C):
        super().__init__()
        self.wR = nn.Parameter(torch.randn(input_dim_R) * 0.01)
        self.wC = nn.Parameter(torch.randn(input_dim_C) * 0.01)
        self.bias_R = nn.Parameter(torch.zeros(1))
        self.bias_C = nn.Parameter(torch.zeros(1))

    def forward(self, xR, xC):
        r = F.linear(xR, self.wR, self.bias_R)
        c = F.linear(xC, self.wC, self.bias_C)

        # Biology-inspired nonlinear modulator
        A = r * (0.5 + torch.sigmoid(2 * r * c))
        prob = torch.sigmoid(A)
        return prob

    def compute_pid_loss(self, xR, xC, y, gamma, bins=5):
        Iunq_R, Iunq_C, Ired, Isyn, Hres = estimate_pid_atoms(xR, xC, y, bins)

        G = (gamma['unq_R'] * Iunq_R +
             gamma['unq_C'] * Iunq_C +
             gamma['red']   * Ired +
             gamma['syn']   * Isyn +
             gamma['res']   * Hres)
        return -G  # We minimize loss (negative of info gain)

class InfomorphicLayer(nn.Module):
    def __init__(self, num_neurons, input_dim_R, input_dim_C):
        super().__init__()
        self.num_neurons = num_neurons
        self.wR = nn.Parameter(torch.randn(num_neurons, input_dim_R) * 0.01)
        self.wC = nn.Parameter(torch.randn(num_neurons, input_dim_C) * 0.01)
        self.bias_R = nn.Parameter(torch.zeros(num_neurons))
        self.bias_C = nn.Parameter(torch.zeros(num_neurons))

    def forward(self, xR, xC):
        """
        xR: (batch_size, input_dim_R)
        xC: (batch_size, input_dim_C)
        Returns:
            prob: (batch_size, num_neurons)
        """
        r = F.linear(xR, self.wR) + self.bias_R  # shape: (batch, num_neurons)
        c = F.linear(xC, self.wC) + self.bias_C

        A = r * (0.5 + torch.sigmoid(2 * r * c))
        prob = torch.sigmoid(A)
        return prob  # Probabilities of HIGH for each neuron

    def compute_total_loss(self, xR, xC, y_batch, gamma, bins=5):
        # y_batch: (batch, num_neurons)
        total_loss = 0
        for i in range(self.num_neurons):
            loss = self.compute_neuron_pid_loss(
                xR.detach(), xC.detach(), y_batch[:, i].detach(), gamma, bins
            )
            total_loss += loss
        return total_loss / self.num_neurons
    
    def compute_neuron_pid_loss(self, xR, xC, y, gamma, bins=5):
        Iunq_R, Iunq_C, Ired, Isyn, Hres = estimate_pid_atoms(xR, xC, y, bins)

        G = (gamma['unq_R'] * Iunq_R +
             gamma['unq_C'] * Iunq_C +
             gamma['red']   * Ired +
             gamma['syn']   * Isyn +
             gamma['res']   * Hres)
        return -G  # We minimize loss (negative of info gain)
# ---------------- Example Training Loop ----------------

class InfomorphicTransformerBlock(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dim, num_neurons):
        super().__init__()
        self.attn = InfomorphicLayer(num_neurons, input_dim, context_dim)
        self.ffn = InfomorphicLayer(num_neurons, num_neurons, context_dim)
        self.out_proj = nn.Linear(num_neurons, tokenizer.vocab_size)

    def forward(self, xR, xC):
        attn_output = self.attn(xR, xC)  # (batch, neurons)
        ffn_output = self.ffn(attn_output, xC)
        return self.out_proj(ffn_output)

class InfomorphicTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_neurons):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.block = InfomorphicTransformerBlock(embed_dim, embed_dim, hidden_dim, num_neurons)

    def forward(self, input_ids):
        xR = self.embed(input_ids)  # (batch, seq, embed)
        context = xR.mean(dim=1)    # crude context
        logits = []
        for t in range(xR.shape[1]):
            out = self.block(xR[:, t], context)
            logits.append(out.unsqueeze(1))
        return torch.cat(logits, dim=1)  # (batch, seq, vocab)



########################################
# 5. Training Loop
########################################


# === Utilities ===
def flatten_model_weights(model):
    return torch.cat([p.data.flatten() for p in model.parameters() if p.requires_grad])


def apply_offset_attn_delta(model, base_weights, delta, batch_idx, max_update=8192, kernel_size=5):
    """
    Vectorized: Applies a soft-local delta around a sliding offset.
    """
    flat_dim = base_weights.numel()
    updated = base_weights.clone()

    # Define offset
    offset = (batch_idx * max_update) % flat_dim
    center_start = offset
    center_end = min(offset + max_update, flat_dim)

    # Clip delta size to match available range
    clipped_delta = delta[:center_end - center_start]

    # Apply primary delta
    updated[center_start:center_end] += clipped_delta

    # Optional: smooth attenuated outer window
    for i in range(1, kernel_size + 1):
        decay = 1.0 / math.sqrt(i + 1)

        left_idx = max(0, center_start - i)
        right_idx = min(flat_dim, center_end + i)

        if left_idx < center_start:
            updated[left_idx:center_start] += decay * clipped_delta[:center_start - left_idx]
        if center_end < right_idx:
            updated[center_end:right_idx] += decay * clipped_delta[:right_idx - center_end]

    # Write back
    load_flat_weights(model, updated)
    print("delta shape:", delta.shape)

def load_flat_weights(model, flat_vector):
    offset = 0
    total = flat_vector.numel()

    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            numel = param.numel()
            if offset + numel > total:
                break  # don't overflow
            slice_ = flat_vector[offset:offset+numel]
            param.data.copy_(slice_.view_as(param))
            offset += numel

def force_move_buffers(model, device):
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            setattr(model, name, buffer.to(device))

def summarize_attention(model):
    fwd_summary = []
    bwd_summary = []
    for block in model.blocks:
        if hasattr(block, 'last_fwd_heads') and block.last_fwd_heads is not None:
            fwd_summary.append(block.last_fwd_heads.mean().unsqueeze(0))
        if hasattr(block, 'last_bwd_heads') and block.last_bwd_heads is not None:
            bwd_summary.append(block.last_bwd_heads.mean().unsqueeze(0))
    return torch.cat(fwd_summary + bwd_summary, dim=0).view(1, -1)

# Then inside training loop:
#summary_input = summarize_attention(model)
#flat_input = summary_input.detach()



def prepare_decoder_input_and_target(target):
    """
    Prepares inputs and targets for teacher forcing when <BOS> is auto-generated by the tokenizer.
    - target: Tensor of shape (batch_size, seq_len)
    Returns:
    - decoder_input: Shifted target, including <BOS>
    - target_output: Original target
    """
    # Shift target to the right to form the decoder input
    decoder_input = torch.zeros_like(target)
    decoder_input[:, 1:] = target[:, :-1]  # Shift right
    decoder_input[:, 0] = target[:, 0]     # Copy the <BOS> from the target

    # The output is the target sequence itself (including <EOS>)
    target_output = target
    
    return decoder_input, target_output


def build_custom_validation_batch(tokenizer, seq_len=seq_len, device=device):
    query_strings = [
        "1. What is 17 + 35?",
        "2. Solve for x: 2x + 5 = 13",
        "3. What is the derivative of x^2?",
        "4. What is the integral of x dx?",
        "5. What is the plural of 'analysis'?",
        "6. Is this sentence correct? 'He go to school every day.'",
        "7. What is the first law of Robotics?",
        "8. What is the secpnd law of robotics?",
        "9. What is the third law of robotics?,",
        "10. What is the zeroth law of robotics?",
        "11. What does this Python function return? def square(x): return x * x",
        "12. Write a function in Python that checks if a number is prime.",
        "13. What is the derivative of a function x^3 according to calculus?",
        "14. Describe the integral of a function x^3 according to calculus, please."
    ]

    target_strings = [
        "1. 52",
        "2. x = 4",
        "3. 2x",
        "4. (1/2)x^2 + C",
        "5. analyses",
        "6. No, it should be: 'He goes to school every day.'",
        "7. 1. A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
        "8. 2. A robot must obey orders given by humans except where such orders would conflict with the First Law.",
        "9. 3. A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.",
        "10. 0. A robot may not harm humanity, or, by inaction, allow humanity to come to harm.",
        "11. It returns the square of x.",
        "12. def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "13. The derivative of x^3 by the power law for derivatives would be 3x^2.",
        "14. According to the integral power law the integral of x^3 would be (x^2)/2."
    ]

    input_ids, target_ids = [], []
    for query, target in zip(query_strings, target_strings):
        q_ids = tokenizer.encode(query, max_length=seq_len, truncation=True, padding='max_length')
        a_ids = tokenizer.encode(target, max_length=seq_len, truncation=True, padding='max_length')

        input_ids.append(q_ids)
        target_ids.append(a_ids)

    return input_ids, target_ids

def train_model(model, dataloader, val_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0

    for batch_idx, (src, target) in enumerate(dataloader):
        
        src = src.to(device)
        target = target.to(device)
        decoder_input, target_labels = prepare_decoder_input_and_target(target)
        decoder_input = decoder_input.to(device)
        target_labels = target_labels.to(device)
        optimizer.zero_grad()
        model = model.to(device)
        # 🔹 Get predictions & rule-modified embeddings
        #output = model(src, decoder_input).to(device)
        output = model(src)
        # 🔹 Ensure `output` and `target_labels` have the same sequence length
        seq_len = min(output.shape[1], target_labels.shape[1])  # Get the shorter sequence length
        output = output[:, :seq_len, :]  # Truncate logits if too long
        target_labels = target_labels[:, :seq_len]  # Truncate targets if too long

        # 🔹 Flatten for cross_entropy()
        loss = criterion(output.reshape(-1, output.shape[-1]), target_labels.reshape(-1))
        n+=1
        print(f"Iteration {n}, Loss: {loss.item()}")
        if torch.isnan(loss) or torch.isinf(loss):
            print("🚨 Warning: NaN or Inf detected in loss! Skipping update.")
            return

        loss.backward()

        # 🔹 Track how rules affected loss
        prev_loss = loss.item()
        # Clip gradients to prevent exploding values
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        try:
                val_src, val_tgt = next(val_loader_iter)
        except:
                val_loader_iter = iter(val_loader)
                val_src, val_tgt = next(val_loader_iter)
        val_src, val_tgt = val_src.to(device), val_tgt.to(device)
        #val_dec_input, val_target = prepare_decoder_input_and_target(val_target)

        val_preds = model(val_src)
        val_loss = criterion(val_preds.view(-1, val_preds.size(-1)), val_tgt.view(-1)).item()

        print(f"Iteration {n}, Validation Loss: {val_loss}")
          
        optimizer.step()                    



        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_hyperplastic(model, hyper, dataloader, val_loader, loss_fn, hyper_opt, opt, device):
    model.train()
    hyper.train()
    total_loss = 0.0
    n=0
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        batch, seq = src.size()

        # Save weights
        w_before = flatten_model_weights(model).detach()

        # Prepare input for hypertransformer
        # Forward pass with updated model
        preds = model(src, direction="forward")
        loss = loss_fn(preds.view(-1, preds.size(-1)), tgt.view(-1))
        n+=1
        print(f"Iteration {n}, Loss: {loss.item()}")

        #memory blocks for later testing
        # Assume model.blocks stores activations
        forward_summary = []
        backward_summary = []
        f=0
        bac=0
        for block in model.blocks:
            if hasattr(block, 'last_fwd_heads'):
                forward_summary.append(block.last_fwd_heads)
                f+=1
            if hasattr(block, 'last_bwd_heads'):
                backward_summary.append(block.last_bwd_heads)
                bac+=1
        if forward_summary and backward_summary:
            flat_input = torch.cat(forward_summary + backward_summary, dim=0)
            #print(f"summary shape: {flat_input.shape} iteration forward: {f} backward {bac}")
            b, d, s, d = flat_input.size()
            flat_input = flat_input.reshape(b, s, d*d) # (b, e/h, s, e/h) -> (b,s*e) *seq_len must equal embed-size
            # Project to match forward weight shape
            #print(flat_input.shape)
            flat_input = flat_input.mean(dim=(1)).unsqueeze(0)  # crude average [1, batch_size * (f+b), embed_size]
            #print(flat_input.shape)
            fb = int(f+bac)
            flat_input = flat_input.reshape(1, batch, (fb), d*d)
            #print(flat_input.shape)
            flat_input = flat_input.mean(dim=(1))  # crude average [1, 2, embed_size]
            #rint(flat_input.shape)
            #print("summaries success")
        
        else:
            with torch.no_grad():
                src_embed = model.embed(src).mean(dim=1)  # (B, C)
                tgt_embed = model.embed(tgt).mean(dim=1)  # (B, C)

        #####OR difference of means
        #concatenation
            #flat_input = torch.cat([src_embed, tgt_embed], dim=1).unsqueeze(0).float()  # (1, 2C)

        # Input vector = [src, tgt, diff]
                flat_input = torch.cat([src_embed, (tgt_embed - src_embed)], dim=0).unsqueeze(0)
                #print(flat_input.shape)

        # Predict delta weights
        delta = hyper(flat_input.to(device)).squeeze(0) # flat_input should be 1, batch_size, embed_size*2
        ##oldload_flat_weights(model, w_before + delta)
        apply_offset_attn_delta(model, w_before, delta, batch_idx, max_update=8192, kernel_size=5)


        # Validation pass to assess performance
        with torch.no_grad():
            try:
                val_src, val_tgt = next(val_loader_iter)
            except:
                val_loader_iter = iter(val_loader)
                val_src, val_tgt = next(val_loader_iter)
            val_src, val_tgt = val_src.to(device), val_tgt.to(device)
            val_preds = model(val_src, direction="forward")
            val_loss = loss_fn(val_preds.view(-1, val_preds.size(-1)), val_tgt.view(-1)).item()

        print(f"Iteration {n}, Validation Loss: {val_loss}")

        # Train hypertransformer to reduce val loss
        hyper_opt.zero_grad()
        loss.backward()
        hyper_opt.step()

        # Backward plasticity for W_fwd update
        opt.zero_grad()
        preds = model(tgt, direction="backward")
        loss = loss_fn(preds.view(-1, preds.size(-1)), src.view(-1))
        print(f"Iteration {n}, Backward_Loss: {loss.item()}")
        opt.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


########################################
#6. inference
########################################


def load_json_file(file_path):
    """Load the JSON dataset file properly."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)  # 🔹 Ensure it's properly parsed
            if not isinstance(data, list):
                raise ValueError("🚨 Loaded data is not a list of dictionaries.")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"🚨 Failed to parse JSON: {e}")

def generate_2(model, prompt, tokenizer, seq_len, device, max_generated=120, repetition_penalty=1.2, top_p=0.9):
    model.eval()
    generated_tokens = []
    model.to(device)
    with torch.no_grad():
        # Tokenize prompt → fixed encoder input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        encoder_input_len = input_ids.size(1)

        # Pad encoder input to max model length
        if encoder_input_len < seq_len:
            pad_len = seq_len - encoder_input_len
            pad_token_id = tokenizer.pad_token_id or 0
            padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long).to(device)
            input_ids = torch.cat([input_ids, padding], dim=1)
        else:
            input_ids = input_ids[:, :seq_len]

        # Encoder is static throughout generation
        encoder_input_ids = input_ids.to(device)

        # Setup initial decoder input
        bos_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id or 0
        tgt_ids = torch.tensor([[bos_token_id]], device=device)

        for _ in range(max_generated):
            # Forward pass through model
            encoder_input_ids.to(device)
            tgt_ids.to(device)
            model.to(device)
            outputs = model(encoder_input_ids, tgt_ids).to(device)
            logits = outputs[:, -1, :]  # (batch, vocab)

            # Repetition penalty
            for token in set(tgt_ids[0].tolist()):
                if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
                    logits[0, token] /= repetition_penalty

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            filtered_logits = logits.clone()
            filtered_logits[0, sorted_indices[0][sorted_indices_to_remove[0]]] = float('-inf')

            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            # Stop at EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Append and continue
            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
            generated_tokens.append(next_token_id.item())

            # Pad if needed to align with model
            if tgt_ids.size(1) > seq_len:
                tgt_ids = tgt_ids[:, -seq_len:]

    return tokenizer.decode(generated_tokens)


def generate_4(model, prompt, tokenizer, seq_len, device, max_generated=120, repetition_penalty=1.2, top_p=0.9):
    model.eval()
    generated_tokens = []
    model.to(device)
    with torch.no_grad():
        # Tokenize prompt → fixed encoder input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        encoder_input_len = input_ids.size(1)

        # Pad encoder input to max model length
        if encoder_input_len < seq_len:
            pad_len = seq_len - encoder_input_len
            pad_token_id = tokenizer.pad_token_id or 0
            padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long).to(device)
            input_ids = torch.cat([input_ids, padding], dim=1)
        else:
            input_ids = input_ids[:, :seq_len]

        # Encoder is static throughout generation
        encoder_input_ids = input_ids.to(device)

        # Setup initial decoder input
        bos_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id or 0
        tgt_ids = torch.tensor([[bos_token_id]], device=device)

        for _ in range(max_generated):
            # Forward pass through model
            encoder_input_ids.to(device)
            tgt_ids.to(device)
            model.to(device)
            outputs = model(encoder_input_ids, direction="forward").to(device)
            logits = outputs[:, -1, :]  # (batch, vocab)

            # Repetition penalty
            for token in set(tgt_ids[0].tolist()):
                if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
                    logits[0, token] /= repetition_penalty

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            filtered_logits = logits.clone()
            filtered_logits[0, sorted_indices[0][sorted_indices_to_remove[0]]] = float('-inf')

            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            # Stop at EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Append and continue
            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
            generated_tokens.append(next_token_id.item())

            # Pad if needed to align with model
            if tgt_ids.size(1) > seq_len:
                tgt_ids = tgt_ids[:, -seq_len:]

    return tokenizer.decode(generated_tokens)
########################################
# 7. Main Function
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=r"C:\Users\Austin\.cursor\ruletransformer-main\mhlatest-main\data", help='Path to JSON data')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=14, help='Batch size for training')
    parser.add_argument('--max_seq_length', type=int, default=seq_len, help='Fixed maximum sequence length')

    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    

    # ***** NEW: Load tokenizer from file instead of building from the data *****

    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    # Load dataset correctly
    #json_data = load_json_file(args.data)

    # Pass parsed JSON instead of raw file path
    data = load_dataset(args.data)
    inputs, targets = tokenize_data(data)
    dataset = prepare_batch(inputs, targets, args.max_seq_length)
    val_inputs, val_targets = build_custom_validation_batch(tokenizer)
    val_dataset = prepare_batch(val_inputs, val_targets, args.max_seq_length)
    # 🔹 Ensure dataset isn't empty
    if len(dataset) == 0:
        raise ValueError("🚨 Dataset is empty after filtering invalid entries! Check your dataset.")

    # Use a lambda to pass the fixed length to collate_fn.
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch))
    seq_length = args.max_seq_length
    d_model = 100
    hidden_d = 128
    n_layers = 6
    n_heads = 10
    n_neurons = 128

    torch.set_printoptions(sci_mode=False)
    torch.set_default_dtype(torch.float32)
    torch.cuda.empty_cache()  # if using CUDA

    # Replace model initialization
    #model = PlasticDualTransformer(vocab_size, d_model, seq_len, n_layers, n_heads).to(device)
    model = InfomorphicTextGenerator(vocab_size, embed_dim=d_model, hidden_dim=hidden_d, num_neurons=n_neurons).to(device)

    #hyper_model = HyperTransformer(input_dim=d_model, model_dim=d_model, num_layers=n_layers, num_heads=n_heads, weight_dim=8192).to(device)
    #hyper_optimizer = optim.AdamW(hyper_model.parameters(), lr=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    

    for buffer_name, buffer in model.named_buffers():
        setattr(model, buffer_name, buffer.to(device))

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_model(model, dataloader, val_loader, optimizer, criterion, device)

        #avg_loss = train_hyperplastic(model, hyper_model, dataloader, val_loader, criterion, hyper_optimizer, optimizer, device)


        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")
    
    
    # Set the model to evaluation mode and perform inference.
    prompt = "What is the critical temperature of a superconducting thin film made of lead with a thickness of 100 nm?"

    #generated_text = generate_2(model,prompt, base_tokenizer, seq_length, device)
    generated_text = generate_4(model,prompt, base_tokenizer, seq_length, device)

    print("Generated text:")
    print(generated_text)

if __name__ == '__main__':
    main()




