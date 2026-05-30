import torch
import torch.nn as nn
import torch.nn.functional as F
# === 新增：日志相关 ===
import time
from datetime import datetime

# Decoder-only Transformer

batch_size = 64
block_size = 256
n_embd = 384
num_heads = 6
head_size = n_embd // num_heads
n_layer = 6
droput_p = 0.2          # 抑制网络加深后的过拟合
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# -------------

torch.manual_seed(1337)

with open('input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(s for s in text)))
vocab_size = len(chars)

stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}

encode = lambda x : [stoi[s] for s in x]
decode = lambda x : ''.join([itos[i] for i in x])

data = torch.tensor(encode(text), dtype=torch.long)
n1 = int(0.9*len(data))
train_data = data[:n1]
val_data = data[n1:]

# === 新增：日志文件初始化（按时间戳命名，避免覆盖历史训练） ===
LOG_PATH = f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
_log_file = open(LOG_PATH, 'w', encoding='utf-8')

def log(msg=''):
    """同时打印到终端并追加到日志文件"""
    print(msg)
    _log_file.write(str(msg) + '\n')
    _log_file.flush()   # 立即写盘，防止训练中断丢失

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x_batch = torch.stack([data[i:i+block_size] for i in ix], dim=0)
    y_batch = torch.stack([data[i+1:i+block_size+1] for i in ix], dim=0)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)   # 数据一开始就放在device上
    return x_batch, y_batch

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:      # 返回两种数据集上的loss
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    # 1. Implement Masked(because it's decoder) "Scaled Dot-Product Attention"（单头的 自(decoder) 注意力）
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(droput_p)
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)
        
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -float('inf'))        # self.tril 是固定大小矩阵。但实际输入序列长度 T 可能小于 block_size（generate时），所以为了适配需要有[:T, :T]的T×T切片
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)     # 随机阻碍某些token之间的通信

        v = self.value(x)
        out = wei @ v    # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    # 2. Implement Masked Multi-Head Attention
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))  # 并行、独立计算的h个头
        self.proj = nn.Linear(num_heads*head_size, n_embd)      # 其实就是(n_embd, n_embd)的全连接层, 让各个头融合通信
        self.dropout = nn.Dropout(droput_p)

    def forward(self, x):
        h_outs = [h(x) for h in self.heads]
        out = torch.cat(h_outs, dim=-1)     # num_heads * (B, T, head_size) -> (B, T, num_heads*head_size=n_embd)
        out = self.dropout( self.proj(out) )
        return out

class FeedForward(nn.Module):
    # 3. Implement Feed Forward Network
    # FFN 是 token 拿到 attention 聚合的信息后"再私下自己（没有跨 token 的信息流动）深度思考"的环节，并且它本质就是一个 MLP
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),      # 升维把每个 token 投到一个更高维的空间里，让非线性激活能"切"出更复杂的特征边界，然后再压缩回原维度。
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(droput_p)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # 4. 有了attention和ffn之后可以构建成一个block，这是trm的可堆叠计算层
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa_heads = MultiHeadAttention(num_heads, head_size)
        self.ffn = FeedForward(n_embd)
        # LayerNorm层里也有各自可训练的γ(=1)、β(=0)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # LayerNorm采用Pre-LN的方式
        x = x + self.sa_heads(self.ln1(x)) # [B, T, C] - Communication Task
        # skip cross attention
        x = x + self.ffn(self.ln2(x))            # Computation Task
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 结合transformer.png看
        # lookup table
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)  # 嵌入表1 - 根据idx对token的身份（索引）进行编码
        self.position_embedding_table = nn.Embedding(block_size, n_embd)    # 嵌入表2 - 根据token在time上的位置进行位置编码
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layer)])        # 堆叠n_layer个block
        self.final_ln = nn.LayerNorm(n_embd)  # 由于采用了Pre-LN的架构，Pre-LN 的残差流在多层累加后会幅值膨胀、分布漂移，最终 LayerNorm 就是把累积膨胀的 x 重新归一化，而 final_ln 让送给 lm_head 的输入永远在合理分布
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size)       # 任务头 - 作为next token概率映射表

    def forward(self, idx, target=None):   # idx和target都是(B, T)的整数index tensor
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)   # [B, T] -> [B, T, C]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # [T, C]（下面会在B上广播）
        x = tok_emb + pos_emb   # [B, T, C] - x不仅包含了token的身份信息，还融合了token的位置信息
        x = self.blocks(x)      # [B, T, C]
        x = self.final_ln(x)    # [B, T, C]
        logits = self.lm_head(x)      # -> [B, T, vocab_size]

        B, T, C = logits.shape
        if target is None:   # 不需要算loss
            loss = None
        else:
            logits = logits.view(B*T, C)
            targets = target.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):   # idx是表示成当前context的(B, T)整数tensor
        # nn.Embedding 只在乎 idx 里的值 是否 ∈ [0, num_embeddings)，和 idx 的形状/长度毫无关系。
        # bigram 里 token_embedding_table 不报错，是因为 idx 拼多长里面的值都还是合法字符 id；
        # 引入 pos_emb 之后会报错，是因为 arange(T) 的最大值会随 T 增长，T 一超过 block_size 就有非法值
        # ——本质从来不是"idx 太长"，而是"idx 里出现了非法值"。
        # 🌶🌶 这也造成了 context window 的硬约束本质
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]         # 模型只能看到（最多）最近 block_size 个 token
                logits, _ = self.forward(idx_cond)
                cur_time_logits = logits[:, -1, :]
                probs = F.softmax(cur_time_logits, dim=-1)
                next_idx = torch.multinomial(probs, 1)
                idx = torch.cat((idx, next_idx), dim=1)
        return idx      # idx 的 T 维度从 t=T 经generate延展成 t=T+max_new_tokens 的 new idx然后输出
    
model = BigramLanguageModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# === 新增：训练开始的元信息记录 ===
log("=" * 60)
log(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Log file: {LOG_PATH}")
log(f"Device: {device}")
log(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")
log("Hyperparameters:")
log(f"  batch_size = {batch_size}")
log(f"  block_size = {block_size}")
log(f"  n_embd     = {n_embd}")
log(f"  num_heads  = {num_heads}")
log(f"  n_layer    = {n_layer}")
log(f"  dropout    = {droput_p}")
log(f"  max_iters  = {max_iters}")
log(f"  lr         = {learning_rate}")
log("-" * 60)

train_start = time.perf_counter()   # 训练计时起点
# 执行训练
for iter in range(max_iters + 1):
    if iter % eval_interval == 0:
        out = estimate_loss(model)
        # === 替换原 print 为 log，并附加累计耗时 ===
        elapsed = time.perf_counter() - train_start
        log(f"step {iter:5d} | train loss {out['train']:.4f} | val loss {out['val']:.4f} | elapsed {elapsed:7.1f}s")
    
    xb, yb = get_batch('train')
    logits, loss = model.forward(xb, yb)

    optimizer.zero_grad(set_to_none=False)
    loss.backward()
    optimizer.step()

# === 新增：训练结束统计 ===
total_time = time.perf_counter() - train_start
log("-" * 60)
log(f"Training finished. Total time: {total_time:.1f}s  ({total_time/60:.2f} min)")
log("=" * 60)

# 用模型进行 generate
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(model.generate(context, max_new_tokens=10000)[0].tolist())

# === 替换原 print 为 log ===
log("Generated sample:")
log(generated)

_log_file.close()   # 关闭日志文件