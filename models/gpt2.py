import argparse
import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, t, c = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * c ** -0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """communication followed by computation"""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT2LanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(t, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            _, _, c = logits.shape
            logits = logits.view(-1, c)
            loss = F.cross_entropy(logits, targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer', action='store_true', help='Whether to train the model')
    parser.add_argument('--sampler', action='store_true', help='Whether to sample from the trained model')
    parser.add_argument('--path', type=str, default='', help='Path to load the trained model')
    parser.add_argument('--max_tokens', type=int, default=500, help='Maximum number of tokens to generate')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--block_size', type=int, default=32, help='Maximum context length')
    parser.add_argument('--max_iters', type=int, default=5000, help='Maximum number of iterations for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='4', help='Device, or -1 for CPU')
    parser.add_argument('--checkpoint', type=int, default=0, help='Interval for saving checkpoints')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--show', type=int, default=1, help='Interval to show training loss')
    parser.add_argument('--SAVE_PATH', type=str, default='./LongtextResults/gpt2', help='Path to save results')
    parser.add_argument('--train_data', type=str, default='./data/dialogues/train.txt')
    parser.add_argument('--test_data', type=str, default='./data/dialogues/test.txt')
    parser.add_argument('--eval_iters', type=int, default=1)
    parser.add_argument('--n_embd', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    args = parser.parse_args()
    if args.n_embd <= 0 or args.n_head <= 0 or args.n_embd % args.n_head:
        parser.error('n_embd must be positive and divisible by n_head')
    if args.bs <= 0 or args.block_size <= 0 or args.max_iters < 0 or args.eval_iters <= 0 or args.show <= 0:
        parser.error('bs, block_size, eval_iters and show must be positive; max_iters must be nonnegative')
    if args.device in ('-1', 'cpu'):
        device = 'cpu'
    elif args.device.startswith('cuda:'):
        device = args.device
    else:
        device = f'cuda:{args.device}'
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return args, device, dtype


def load_data(args, device):
    with open(args.train_data, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_text = f.read().lower()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    train_data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    val_data = torch.tensor([stoi[c] for c in test_text], dtype=torch.long)
    if min(len(train_data), len(val_data)) <= args.block_size:
        raise ValueError('Training and test text must be longer than block_size')
    return train_data, val_data, stoi, itos


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device):
    model.eval()
    out = {}
    for name, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.bs, args.block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean()
    model.train()
    return out


def save_info(args, vocab_size):
    os.makedirs(args.SAVE_PATH, exist_ok=True)
    with open(os.path.join(args.SAVE_PATH, 'info.toml'), 'w') as f:
        f.write('[model_info]\n')
        f.write('model = "gpt2"\n')
        f.write(f'block_size = {args.block_size}\n')
        f.write(f'batch_size = {args.bs}\n')
        f.write(f'max_iters = {args.max_iters}\n')
        f.write(f'lr = {args.lr}\n')
        f.write(f'seed = {args.seed}\n')
        f.write(f'vocab_size = {vocab_size}\n')
        f.write(f'n_embd = {args.n_embd}\n')
        f.write(f'n_head = {args.n_head}\n')
        f.write(f'n_layer = {args.n_layer}\n')
        f.write(f'dropout = {args.dropout}\n')
        f.write(f'dtype = "{args.dtype}"\n')
        f.write(f'eval_iters = {args.eval_iters}\n')
        f.write(f'checkpoint = {args.checkpoint}\n')
        f.write(f'max_tokens = {args.max_tokens}\n')
        f.write(f'train_data = {json.dumps(args.train_data)}\n')
        f.write(f'test_data = {json.dumps(args.test_data)}\n')


def make_model(args, vocab_size, device, dtype):
    return GPT2LanguageModel(
        vocab_size=vocab_size, block_size=args.block_size,
        n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer,
        dropout=args.dropout
    ).to(device=device, dtype=dtype)


def train(args, device, dtype):
    existing_files = set(os.listdir(args.SAVE_PATH)) if os.path.exists(args.SAVE_PATH) else set()
    if existing_files - {'info.toml', 'vocab.json'}:
        raise FileExistsError(f'Result directory is not empty: {args.SAVE_PATH}')
    train_data, val_data, stoi, itos = load_data(args, device)
    save_info(args, len(stoi))
    with open(os.path.join(args.SAVE_PATH, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(list(stoi), f, ensure_ascii=True)
    model = make_model(args, len(stoi), device, dtype)
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loss, val_loss = [], []
    if args.checkpoint > 0:
        os.makedirs(os.path.join(args.SAVE_PATH, 'checkpoints'), exist_ok=True)
    for iteration in range(args.max_iters):
        losses = estimate_loss(model, train_data, val_data, args, device)
        if iteration % args.show == 0:
            print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        train_loss.append(losses['train'].item())
        val_loss.append(losses['val'].item())
        xb, yb = get_batch(train_data, args.bs, args.block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if args.checkpoint > 0 and iteration % args.checkpoint == 0 and iteration > 0:
            torch.save(model.state_dict(), os.path.join(args.SAVE_PATH, 'checkpoints', f'iter_{iteration}.pt'))
    torch.save(model.state_dict(), os.path.join(args.SAVE_PATH, 'model_final.pth'))
    np.savetxt(os.path.join(args.SAVE_PATH, 'train_record.txt'), np.array(train_loss))
    np.savetxt(os.path.join(args.SAVE_PATH, 'val_record.txt'), np.array(val_loss))
    if train_loss:
        train_record = torch.tensor(train_loss)
        val_record = torch.tensor(val_loss)
        print(f'Train Loss: {train_record[-100:].mean()} +- {train_record[-100:].std()}\n')
        print(f'Validation Loss: {val_record[-100:].mean()} +- {val_record[-100:].std()}\n')
    sample_rng = {'cpu': torch.get_rng_state()}
    if str(device).startswith('cuda'):
        sample_rng['cuda'] = torch.cuda.get_rng_state(device)
    torch.save(sample_rng, os.path.join(args.SAVE_PATH, 'sample_rng.pth'))
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=args.max_tokens)[0].tolist()
    generated_text = ''.join(itos[i] for i in generated)
    print(generated_text)
    with open(os.path.join(args.SAVE_PATH, f'generated_text_seed{args.seed}.txt'), 'w', encoding='utf-8') as f:
        f.write(generated_text)
    np.savetxt(os.path.join(args.SAVE_PATH, f'generated_data_seed{args.seed}.txt'), np.array(generated), fmt='%d')
    return model


def load_model(args, device, dtype):
    import tomllib
    path = args.path or args.SAVE_PATH
    with open(os.path.join(path, 'info.toml'), 'rb') as f:
        info = tomllib.load(f)['model_info']
    for key in ('block_size', 'n_embd', 'n_head', 'n_layer', 'dropout'):
        setattr(args, key, info[key])
    dtype = torch.float32 if info.get('dtype', 'float32') == 'float32' else torch.float64
    model = make_model(args, info['vocab_size'], device, dtype)
    model.load_state_dict(torch.load(os.path.join(path, 'model_final.pth'), map_location=device))
    return model, path, info


def sample(args, device, dtype):
    model, path, info = load_model(args, device, dtype)
    with open(os.path.join(path, 'vocab.json'), 'r', encoding='utf-8') as f:
        chars = json.load(f)
    sample_rng = torch.load(os.path.join(path, 'sample_rng.pth'), map_location='cpu')
    torch.set_rng_state(sample_rng['cpu'])
    if str(device).startswith('cuda') and 'cuda' in sample_rng:
        torch.cuda.set_rng_state(sample_rng['cuda'], device)
    # The notebook generates in train mode, including dropout when enabled.
    model.train()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=args.max_tokens)[0].tolist()
    generated_text = ''.join(chars[i] for i in generated)
    print(generated_text)
    with open(os.path.join(path, f'generated_text_seed{args.seed}.txt'), 'w', encoding='utf-8') as f:
        f.write(generated_text)
    np.savetxt(os.path.join(path, f'generated_data_seed{args.seed}.txt'), np.array(generated), fmt='%d')


if __name__ == '__main__':
    args, device, dtype = set_args()
    if args.trainer or not args.sampler:
        train(args, device, dtype)
    elif args.sampler:
        sample(args, device, dtype)
