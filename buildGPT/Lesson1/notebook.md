## 目录

- [1. `torch.stack`](#1-torchstack)
- [2. `nn.Embedding`](#2-nnembedding)
- [3. `F.cross_entropy`](#3-fcross_entropy)
- [4. `torch.optim`](#4-torchoptim)
- [5. `model.eval()` 和 `model.train()`](#5-modeleval-和-modeltrain)
- [6. torch 的 `@` 操作（矩阵乘法）](#6-torch-的-操作矩阵乘法)
- [7. `tensor.masked_fill`](#7-tensormasked_fill)
- [8. `nn.Module`](#8-nnmodule)
- [9. `nn.Module`的对象名作为方法（callable object）的机制](#9-nnmodule的对象名作为方法callable-object的机制)
- [10. `nn.Linear`总结](#10-nnlinear总结)
- [11. Self-Attention vs Cross-Attention 精简对照](#11-self-attention-vs-cross-attention-精简对照)
- [12. Scaled Dot-Product Attention 的因果链](#12-scaled-dot-product-attention-的因果链)
- [13. `register_buffer`](#13-register_buffer)
- [14. 可学习位置编码的模型的 context window](#14-可学习位置编码的模型的-context-window)
- [15. `nn.Sequential`](#15-nnsequential)
- [16. LayerNorm](#16-layernorm)
- [17. Pre-LN vs Post-LN](#17-pre-ln-vs-post-ln)

------


#### 1. `torch.stack`

**官方名称：** `torch.stack`
**官方定义：** 沿新维度连接一系列张量。所有张量必须具有相同形状。

**Signature：**

```python
torch.stack(tensors, dim=0, *, out=None)
```

**一句话核心：** 在新维度上“堆叠”张量，输入张量形状相同，输出比输入**多一维**。

---

### 示例

```python
a = torch.tensor([1, 2, 3])  # [3]
b = torch.tensor([4, 5, 6])  # [3]

# 沿 dim=0 堆叠（默认）
torch.stack([a, b], dim=0)   # shape [2, 3]
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# 沿 dim=1 堆叠
torch.stack([a, b], dim=1)   # shape [3, 2]
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
```

---

### 与 `torch.cat` 的区别

| 操作    | 是否增维 | 输入形状  | 输出形状              |
| ------- | -------- | --------- | --------------------- |
| `cat`   | ❌ 不增   | `[3],[3]` | `[6]`（沿已有维拼接） |
| `stack` | ✅ 增一维 | `[3],[3]` | `[2,3]`（新维堆叠）   |

```python
x = torch.tensor([1,2])
y = torch.tensor([3,4])

torch.cat([x, y])      # [1,2,3,4]   shape [4]
torch.stack([x, y])    # [[1,2],[3,4]] shape [2,2]
```

---

### 常见用途

**1. 将多个样本堆成 batch**
```python
sample1 = torch.randn(3, 224, 224)
sample2 = torch.randn(3, 224, 224)
batch = torch.stack([sample1, sample2], dim=0)  # [2, 3, 224, 224]
```

**2. 堆叠多个时间步的特征**
```python
t1 = torch.randn(batch, 256)
t2 = torch.randn(batch, 256)
timesteps = torch.stack([t1, t2], dim=1)  # [batch, 2, 256]
```

---

### 一句话记忆

**stack = 新维度堆叠，shape 加一维；cat = 旧维度拼接，shape 不变。**

-----

#### 2. `nn.Embedding`

**官方名称：** `torch.nn.Embedding`
**官方定义：** 一个简单的查找表（Lookup Table），存储固定大小的词向量（embedding）。给定索引，返回对应的向量。

**Signature：**

```python
nn.Embedding(num_embeddings, embedding_dim, ...)
```

**一句话核心：** 将整数索引（如单词ID）映射为稠密向量，本质上是一个可训练的查表操作。

---

### 示例

```python
import torch.nn as nn

# 创建一个 Embedding 层：词典大小 10，每个词向量维度 4
embedding = nn.Embedding(10, 4)

# 输入：一批索引（2个样本）
indices = torch.tensor([1, 3])   # shape [2]

# 输出：每个索引对应的向量
output = embedding(indices)      # shape [2, 4]
# tensor([[ 0.1234, -0.5678,  0.9012, -0.3456],
#         [ 0.2345, -0.6789,  0.0123, -0.4567]], grad_fn=<EmbeddingBackward>)
```

---

### 参数说明

| 参数             | 说明                           |
| ---------------- | ------------------------------ |
| `num_embeddings` | 词表大小（有多少个不同的索引） |
| `embedding_dim`  | 每个索引映射成的向量维度       |

**底层存储：** `weight` 矩阵，形状 `[num_embeddings, embedding_dim]`，可训练。

---

### 工作原理

```python
# 内部等价于
weight = embedding.weight      # [10, 4]
output = weight[indices]       # 花式索引查表
```

---

### 输入输出形状

| 输入形状           | 输出形状                          |
| ------------------ | --------------------------------- |
| `[batch]`          | `[batch, embedding_dim]`          |
| `[batch, seq_len]` | `[batch, seq_len, embedding_dim]` |
| 任意形状 `[*]`     | `[* , embedding_dim]`             |

```python
# 输入 [2, 3]（2个句子，每句3个词）
indices = torch.tensor([[1,2,3],[4,5,6]])
out = embedding(indices)   # shape [2, 3, 4]
```

---

### 与 `nn.Embedding.from_pretrained()`

用预训练好的向量（如 Word2Vec、GloVe）初始化：
```python
pretrained_weights = torch.randn(10, 4)  # 假设已有
embedding = nn.Embedding.from_pretrained(pretrained_weights)
```

---

### 一句话记忆

**Embedding = 索引→向量的可训练查找表，花式索引 `weight[indices]` 的封装。**

------

#### 3.  `F.cross_entropy`

**官方名称：** `torch.nn.functional.cross_entropy`
**官方定义：** 计算输入 `logits` 与目标 `target` 之间的交叉熵损失。该函数结合了 `log_softmax` 和 `nll_loss`。

**Signature：**

```python
F.cross_entropy(input, target, weight=None, reduction='mean', ...)
```

**一句话核心：** 输入原始 logits（不经过 softmax），内部自动做 `log_softmax + nll_loss`，直接返回损失值。

---

### 示例

```python
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1]])  # [1, 3] 未归一化分数
target = torch.tensor([0])                # 真实类别索引

loss = F.cross_entropy(logits, target)    # tensor(0.417)
```

---

### 内部计算过程

```python
# 等价于
probs = F.softmax(logits, dim=-1)         # [0.659, 0.242, 0.099]
loss = -torch.log(probs[0, target])       # -log(0.659) = 0.417
```

---

### 参数说明

| 参数        | 说明                                                         |
| ----------- | ------------------------------------------------------------ |
| `input`     | 模型输出 logits，形状 `[batch, C, ...]`，**强调C在第二维**   |
| `target`    | 真实标签，两种形式：<br>• 类索引：`[batch, ...]`，每个值在 `[0, C-1]`<br>• one-hot：与 `input` 形状相同 |
| `weight`    | 各类别权重，`[C]`，处理类别不平衡                            |
| `reduction` | `'mean'`（默认）、`'sum'`、`'none'`                          |

---

### 形状规则

```python
# 常见情况：2D logits
logits = torch.randn(32, 10)   # [batch, 类别数]
target = torch.randint(0, 10, (32,))  # [batch]
loss = F.cross_entropy(logits, target)  # 标量

# 高维（如图像分割）
logits = torch.randn(16, 5, 64, 64)  # [batch, 类别, 高, 宽]
target = torch.randint(0, 5, (16, 64, 64))  # [batch, 高, 宽]
loss = F.cross_entropy(logits, target)
```

如果`logits`的`C`维度不在第二维，则需要将其和`target`一起进行`view`：

**view 前**：你需要用"两层循环"才能数到一个计算单元

```
for b in range(B):          # 从上往下数样本
    for t in range(T):      # 从左往右数时间步
        取出 logits[b, t, :]   ← 一个长度为 C 的向量
        配上 target[b, t]      ← 一个标量
```

**view 后**：变成"一层循环"

```
for k in range(B*T):        # 只管从上往下数
    取出 logits_flat[k, :]    ← 还是那个长度为 C 的向量
    配上 target_flat[k]       ← 还是那个标量
```

> **"计算单元"本身（一个 C 维 logits 行 + 一个 scalar target）从来没变过，变的只是我们用几个下标去定位它。**

---

### 与 `nn.CrossEntropyLoss` 的关系

|          | `F.cross_entropy`                      | `nn.CrossEntropyLoss`                                        |
| -------- | -------------------------------------- | ------------------------------------------------------------ |
| 类型     | 函数                                   | 模块（带状态）                                               |
| 调用     | 直接 `F.cross_entropy(logits, target)` | 先 `criterion = nn.CrossEntropyLoss()` 再 `criterion(logits, target)` |
| 使用场景 | 函数式风格                             | 面向对象风格，可放入 `nn.Module` 的 `__init__`               |

---

### 注意事项

1. **输入必须是 logits，不是概率**：不要在传入前手动做 softmax
2. **target 是类索引，不是 one-hot**（多数情况）
3. **数值稳定**：内部实现已做稳定化处理

---

### 一句话记忆

**cross_entropy = log_softmax + nll_loss，喂 logits 出损失。**

---------

#### 4. `torch.optim`

**官方名称：** `torch.optim`
**官方定义：** 一个实现多种优化算法的包，用于根据计算出的梯度更新模型参数。

**一句话核心：** 优化器 = 梯度下降的各种变体，负责告诉模型参数“往哪个方向走、走多大步”。

---

### 基本用法（三步）

```python
import torch.optim as optim

# 1. 创建优化器：传入模型参数和学习率
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 2. 训练循环中：清零梯度、反向传播、更新参数
optimizer.zero_grad()   # 清空上一步的梯度
loss.backward()         # 计算当前梯度
optimizer.step()        # 用梯度更新参数
```

其中`zero_grad()`内置参数`set_to_none`：

| `set_to_none=False`（默认） | 将梯度张量填充为 0（`grad.zero_()`） | 需要写内存         |
| --------------------------- | ------------------------------------ | ------------------ |
| `set_to_none=True`          | 将梯度设置为 `None`                  | 只改指针，不写内存 |

---

### 常用优化器对比

| 优化器      | 一句话特点                     | 典型学习率  |
| ----------- | ------------------------------ | ----------- |
| **SGD**     | 最经典，朴素梯度下降，可带动量 | 0.01 ~ 0.1  |
| **Adam**    | 自适应学习率，收敛快，最常用   | 1e-3 ~ 1e-4 |
| **AdamW**   | Adam + 正确的权重衰减（推荐）  | 1e-3 ~ 1e-4 |
| **RMSprop** | 自适应，适合RNN                | 1e-3 ~ 1e-4 |
| **Adagrad** | 自适应，适合稀疏特征           | 1e-2        |

---

### 各优化器详细用法

**1. SGD（随机梯度下降）**
```python
# 基础版
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 带动量（Nesterov 动量）
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

**2. Adam（最常用）**
```python
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
```

**3. AdamW（推荐，权重衰减更干净）**
```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

**4. RMSprop**
```python
optimizer = optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
```

---

### 常用方法

```python
# 清零梯度（必须做，否则梯度会累加）
optimizer.zero_grad()

# 单步更新参数
optimizer.step()

# 获取当前学习率
current_lr = optimizer.param_groups[0]['lr']

# 动态修改学习率
for param_group in optimizer.param_groups:
    param_group['lr'] = new_lr

# 保存/加载优化器状态（用于恢复训练）
torch.save(optimizer.state_dict(), 'optimizer.pt')
optimizer.load_state_dict(torch.load('optimizer.pt'))
```

---

### 不同参数设置不同学习率

```python
# 为不同层设置不同学习率
optimizer = optim.SGD([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(), 'lr': 1e-3},
], lr=1e-4)  # 默认学习率
```

---

### 一句话记忆

**optimizer = 梯度下降引擎：zero_grad 清零，backward 算梯度，step 更新参数。**

-----------

#### 5. `model.eval()` 和 `model.train()`

**官方名称：** `train()` 和 `eval()`
**官方定义：** 设置模型为训练模式或评估/推理模式，影响某些层（如 Dropout、BatchNorm）的行为。

**Signature：**

```python
model.train()  # 训练模式
model.eval()   # 评估模式
```

---

### 一句话核心

**`train()` 和 `eval()` 是模块的“模式开关”，告诉模型现在是“学习阶段”还是“考试阶段”。**

---

### 两者区别

| 层类型        | `train()` 模式              | `eval()` 模式                  |
| ------------- | --------------------------- | ------------------------------ |
| **Dropout**   | 随机丢弃神经元（激活）      | 不丢弃，保留全部               |
| **BatchNorm** | 使用当前 batch 的均值和方差 | 使用训练时累积的全局均值和方差 |

---

### 示例

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.5)
)

# 训练时
model.train()
output = model(x_train)   # Dropout 生效，BN 用 batch 统计量

# 评估时
model.eval()
with torch.no_grad():     # 通常配合使用
    output = model(x_val) # Dropout 失效，BN 用全局统计量
```

---

### 各层的行为详解

**Dropout：**
- `train()`：以概率 p 随机置零，其余缩放
- `eval()`：恒等映射（什么都不做）

**BatchNorm：**
- `train()`：用当前 batch 计算均值和方差，同时更新全局统计量（running_mean）
- `eval()`：使用训练中累积的 running_mean 和 running_var

---

### 注意事项

1. **`model.eval()` 不自动禁用梯度**，需配合 `@torch.no_grad()` 或 `with torch.no_grad():`
2. 模式只影响设置了相关层（Dropout、BN）的模块，线性层、卷积层不受影响
3. 调用 `train()` 后，Dropout 和 BN 会回到训练行为

---

### 一句话记忆

**`model.train()` = 开 Dropout、BN 用 batch 统计量；`model.eval()` = 关 Dropout、BN 用全局统计量。**

------

#### 6. torch的 `@` 操作（矩阵乘法）

**官方名称：** `torch.matmul` 的运算符形式 `@`
**官方定义：** 执行矩阵乘法（matrix multiplication）。

**一句话核心：** **`@` 不遵循逐元素的广播机制，它遵循的是**矩阵乘法的批处理广播规则**，与 `torch.matmul` 完全相同。**

---

### 关键区别

| 操作               | 广播机制类型                                                 |
| ------------------ | ------------------------------------------------------------ |
| `+`、`-`、`*`、`/` | 逐元素广播（元素级）                                         |
| `@`（矩阵乘）      | **矩阵乘法的批处理广播**（最后两维做矩阵乘，前面维度按规则匹配） |

---

### 矩阵乘法的广播规则

对于两个张量 `A` 和 `B`：
1. 最后两维执行矩阵乘法 `(..., m, n) @ (..., n, p) → (..., m, p)`
2. 前面的维度（batch 维度）遵循标准广播规则

```python
import torch

# 形状规则
A = torch.randn(3, 2, 4)   # [3, 2, 4]
B = torch.randn(3, 4, 5)   # [3, 4, 5]
C = A @ B                  # [3, 2, 5]（batch 维度相同）

# 广播 batch 维度
A = torch.randn(3, 2, 4)   # [3, 2, 4]
B = torch.randn(4, 5)      # [4, 5] → 广播为 [3, 4, 5]
C = A @ B                  # [3, 2, 5]
```

---

### 与逐元素广播的对比

```python
# 逐元素乘法（广播）
x = torch.randn(3, 4)
y = torch.randn(4)         # [4] → [3, 4]
z = x * y                  # ✅ [3, 4]

# 矩阵乘法（不同规则）
x = torch.randn(3, 4)
y = torch.randn(3, 4)      # 最后两维 4 和 4 无法做矩阵乘
z = x @ y.T                # ✅ 需要转置，变成 [3,4] @ [4,3] → [3,3]
```

---

### 常见情况总结

| A 形状           | B 形状           | `A @ B` 形状        | 说明       |
| ---------------- | ---------------- | ------------------- | ---------- |
| `[m, n]`         | `[n, p]`         | `[m, p]`            | 标准矩阵乘 |
| `[batch, m, n]`  | `[batch, n, p]`  | `[batch, m, p]`     | 批量矩阵乘 |
| `[batch, m, n]`  | `[n, p]`         | `[batch, m, p]`     | B 广播     |
| `[m, n]`         | `[batch, n, p]`  | `[batch, m, p]`     | A 广播     |
| `[batch1, m, n]` | `[batch2, n, p]` | 需 batch 维度可广播 | 复杂情况   |

---

### 一句话记忆

**`@` 不是逐元素广播，它是最后两维做矩阵乘，前面维度按广播规则匹配。**

-------

#### ⭐7.  `tensor.masked_fill`

**官方名称：** `masked_fill`
**官方定义：** 将张量中满足掩码条件（`mask` 为 `True`）的位置填充为指定的值 `value`。

**Signature：**
```python
tensor.masked_fill(mask, value)
```

**一句话核心：** 根据布尔掩码，在 `True` 的位置上填入指定值，其他位置保持不变。

---

### 示例

```python
import torch

x = torch.tensor([1, 2, 3, 4])
mask = torch.tensor([True, False, True, False])

x.masked_fill(mask, 0)   # tensor([0, 2, 0, 4])

# 原地版本
x.masked_fill_(mask, 0)  # 直接修改 x
```

---

### 参数说明

| 参数    | 说明                                     |
| ------- | ---------------------------------------- |
| `mask`  | 布尔张量，形状需与 `tensor` 相同或可广播 |
| `value` | 填充值（标量）                           |

---

### 常见用法

**1. 将 NaN 或无穷大替换为 0**
```python
x = torch.tensor([1.0, float('nan'), 3.0, float('inf')])
mask = torch.isnan(x) | torch.isinf(x)
x.masked_fill_(mask, 0)   # tensor([1., 0., 3., 0.])
```

**2. 注意力机制中的掩码（将无效位置设为 -inf）**
```python
# 因果掩码（decoder 自注意力）
attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
attention_scores.masked_fill(attn_mask, float('-inf'))
```

**3. 将低于阈值的值置零**
```python
x = torch.randn(5)
threshold = 0.5
mask = x.abs() < threshold
x.masked_fill_(mask, 0)   # 小值归零
```

**4. 填充序列到相同长度（NLP padding）**
```python
# 假设 padding_idx=0
mask = (input_ids == 0)   # padding 位置为 True
embeddings.masked_fill(mask.unsqueeze(-1), 0)
```

---

### 与类似函数的对比

| 函数                | 行为                                                     |
| ------------------- | -------------------------------------------------------- |
| `masked_fill`       | 掩码为 `True` 的位置填入指定值                           |
| `masked_fill_`      | 原地版本，带下划线                                       |
| `where(cond, a, b)` | 条件 `True` 取 `a`，`False` 取 `b`，更灵活（可填不同值） |

```python
# where 能做到 masked_fill 做不到的事
torch.where(x > 0, x, torch.zeros_like(x))  # 负数变0，正数保留

# masked_fill 只能填单一值
x.masked_fill(x > 0, 1)  # 所有正数变1
```

---

### 注意事项

1. **`value` 必须是标量**，不能是张量
2. **掩码的广播规则**：`mask` 形状必须与 `tensor` 一致，或可广播
3. **原地版本 `_`** 更省内存，但会修改原张量

---

### 一句话记忆

**`masked_fill` = 在掩码 `True` 的位置填同一个数，其他位置不动。**

--------

#### 8.  `nn.Module`

**官方定义：** `nn.Module` 是所有神经网络模块的基类，自定义模型必须继承它。

**一句话核心：** 继承 `nn.Module` 是为了**免费获得 PyTorch 的所有核心功能**：参数管理、训练模式切换、设备迁移、梯度追踪等。

常用重载的方法，**一句话核心：** 绝大多数情况只需重载 `__init__` 和 `forward`，其他方法按需重载。

---

### 必须重载

| 方法       | 用途                       | 是否必须   |
| ---------- | -------------------------- | ---------- |
| `__init__` | 初始化层、注册子模块和参数 | ✅ 几乎必须 |
| `forward`  | 定义前向计算逻辑           | ✅ 必须     |

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)
```

---

### 按需重载（进阶）

| 方法                             | 用途                       | 典型场景               |
| -------------------------------- | -------------------------- | ---------------------- |
| `parameters`                     | 自定义参数收集逻辑         | 需要过滤部分参数不更新 |
| `train` / `eval`                 | 自定义训练/评估行为        | 自定义层的特殊模式切换 |
| `apply`                          | 递归应用函数到所有子模块   | 自定义初始化策略       |
| `add_module`                     | 自定义添加子模块的行为     | 动态命名或包装         |
| `state_dict` / `load_state_dict` | 自定义保存/加载逻辑        | 版本迁移、兼容处理     |
| `to`                             | 自定义设备/数据类型转换    | 特殊属性的跨设备处理   |
| `extra_repr`                     | 自定义 `print(model)` 输出 | 添加额外信息到模型表示 |

---

### 详细说明

#### `parameters()` — 参数过滤
```python
def parameters(self, recurse=True):
    # 只返回需要更新的参数（冻结部分参数）
    for name, param in super().parameters(recurse):
        if 'bias' not in name:  # 只返回权重，不返回偏置
            yield param
```

#### `train()` / `eval()` — 自定义模式切换
```python
def train(self, mode=True):
    super().train(mode)
    if mode:
        self.custom_dropout.enable()
    else:
        self.custom_dropout.disable()
```

#### `apply()` — 自定义初始化
```python
def _init_weights(self):
    def init_fn(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    self.apply(init_fn)
```

#### `extra_repr()` — 定制打印信息
```python
def extra_repr(self):
    return f"dim={self.dim}, dropout={self.dropout}"
# 打印模型时会显示这些信息
```

---

### 大多数代码的实际做法

```python
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()          # ✅ 必须
        self.fc = nn.Linear(input_dim, output_dim)  # 注册子模块
    
    def forward(self, x):           # ✅ 必须
        return self.fc(x)
    
    # 其他方法几乎不重载
```

---

### 一句话记忆

**必重载：`__init__` 和 `forward`；进阶重载：`parameters`、`train/eval`、`apply`、`extra_repr`，其他保持默认。**

----------

#### 9. `nn.Module`的对象名作为方法（callable object）的机制

**一句话核心：** 类实现了 `__call__` 方法后，其实例可以像函数一样被调用。`nn.Module` 的 `__call__` 内部会调用 `forward`。

---

### `nn.Module` 中的调用链

```python
# 用户代码
output = model(x)

# 实际执行链
model(x) → model.__call__(x) → model.forward(x) → 返回结果
```

**关键：** 你重载的是 `forward`，而 `__call__` 是父类 `nn.Module` 已实现好的，**不该重载**。

---

### `__call__` 做了什么（简化版）

```python
class Module:
    def __call__(self, *args, **kwargs):
        # 1. 调用 forward_pre_hook
        # 2. 调用 self.forward(*args, **kwargs)  ← 你重载的那个
        # 3. 调用 forward_hook
        return result
```

---

### 其他类似用法（Python 内置协议）

| 特殊方法                 | 触发时机         | 典型用途                                          |
| ------------------------ | ---------------- | ------------------------------------------------- |
| `__call__`               | `obj()`          | 让实例像函数一样调用（如 `nn.Module`、`loss_fn`） |
| `__getitem__`            | `obj[key]`       | 索引/切片（如 `tensor[0]`、`DataLoader[i]`）      |
| `__setitem__`            | `obj[key] = val` | 赋值（如 `state_dict['key'] = val`）              |
| `__len__`                | `len(obj)`       | 返回长度（如 `len(dataloader)`）                  |
| `__iter__`               | `for x in obj:`  | 可迭代（如 `DataLoader`）                         |
| `__enter__` / `__exit__` | `with obj:`      | 上下文管理（如 `with torch.no_grad():`）          |
| `__add__`                | `obj1 + obj2`    | 运算符重载（如张量加法）                          |
| `__matmul__`             | `obj1 @ obj2`    | `@` 运算符（如矩阵乘法）                          |

---

### 在 PyTorch 中的具体例子

**1. `__call__` — 模型调用**
```python
model = MyModel()
output = model(x)      # 调用 model.__call__(x) → forward(x)
```

**2. `__getitem__` — 张量索引**
```python
x = torch.tensor([1, 2, 3])
x[0]      # 调用 x.__getitem__(0)
```

**3. `__len__` — DataLoader 长度**
```python
loader = DataLoader(dataset, batch_size=32)
len(loader)   # 调用 loader.__len__()
```

**4. `__iter__` — DataLoader 迭代**
```python
for batch in loader:   # 调用 loader.__iter__()
    pass
```

**5. `__enter__` / `__exit__` — 梯度禁用上下文**
```python
with torch.no_grad():  # 调用 __enter__ 和 __exit__
    output = model(x)
```

**6. `__matmul__` — 矩阵乘法**
```python
a @ b   # 调用 a.__matmul__(b)
```

**7. `__add__` — 张量加法**
```python
a + b   # 调用 a.__add__(b)
```

---

### 注意事项

| 方法       | 在 `nn.Module` 中的建议 |
| ---------- | ----------------------- |
| `__call__` | ❌ 不要重载              |
| `forward`  | ✅ 这是你要写逻辑的地方  |

**为什么 `nn.Module` 设计成 `__call__` 调 `forward`？**
- `__call__` 负责“前置/后置钩子”等框架逻辑
- `forward` 只负责用户的计算逻辑
- 分离关注点，确保钩子机制正常工作

---

### 一句话记忆

**对象名加括号 `obj()` 触发 `__call__`，`nn.Module` 的 `__call__` 自动调用你重载的 `forward`。类似用法有索引 `obj[idx]`、长度 `len(obj)`、迭代 `for x in obj` 等。**

----------

#### 10. `nn.Linear`总结

**官方名称：** `torch.nn.Linear`
**官方定义：** 对输入应用线性变换 `y = xW^T + b`

**Signature：**

```python
nn.Linear(in_features, out_features, bias=True)
```

---

### 一句话核心

**`nn.Linear` = 只改变最后一维的全连接层，前面所有维度视为 batch，自动广播处理。**

---

### 核心公式

```python
y = x @ W.T + b
```

| 符号 | 形状                          | 说明                          |
| ---- | ----------------------------- | ----------------------------- |
| `x`  | `[*, in_features]`            | 输入，`*` 表示任意 batch 维度 |
| `W`  | `[out_features, in_features]` | 权重，可训练                  |
| `b`  | `[out_features]`              | 偏置，可训练（可选）          |
| `y`  | `[*, out_features]`           | 输出                          |

---

### 形状变换规则

| 输入形状           | 输出形状            | 变化                    |
| ------------------ | ------------------- | ----------------------- |
| `[in]`             | `[out]`             | 1D → 1D                 |
| `[batch, in]`      | `[batch, out]`      | 2D → 2D                 |
| `[batch, seq, in]` | `[batch, seq, out]` | 3D → 3D（仅改最后一维） |
| `[..., in]`        | `[..., out]`        | 任意形状，只改最后一维  |

---

### 内部计算拆解

```python
# 等价实现
def linear_forward(x, weight, bias):
    # x: [*, in], weight: [out, in], bias: [out]
    original_shape = x.shape
    x_flat = x.view(-1, in_features)        # 展平 batch 维度
    out_flat = x_flat @ weight.T            # 矩阵乘法
    if bias is not None:
        out_flat += bias
    return out_flat.view(*original_shape[:-1], -1)
```

---

### 关键特性

| 特性           | 说明                                                        |
| -------------- | ----------------------------------------------------------- |
| **权重形状**   | `[out_features, in_features]`（注意顺序）                   |
| **batch 无关** | `W` **固定为 2D 矩阵**，与 batch 大小无关（batch agnostic） |
| **广播机制**   | 自动将 `W` 应用到所有 batch 维度                            |
| **可学习参数** | `weight` 和 `bias` 都是 `nn.Parameter`                      |

---

### 常见用法

```python
# 基础用法
fc = nn.Linear(512, 256)

# 2D 输入（标准全连接）
x = torch.randn(32, 512)   # [batch, features]
y = fc(x)                   # [32, 256]

# 3D 输入（NLP/时序）
x = torch.randn(16, 50, 512)  # [batch, seq, features]
y = fc(x)                     # [16, 50, 256]

# 不加偏置
fc = nn.Linear(512, 256, bias=False)
```

---

### 与其他层的关系

| 层             | 操作                           | 适用场景         |
| -------------- | ------------------------------ | ---------------- |
| `nn.Linear`    | 全连接，每个输出与所有输入相连 | 特征变换、分类头 |
| `nn.Conv1d`    | 局部连接，共享权重             | 序列/时序        |
| `nn.Embedding` | 查表                           | 离散索引映射     |

---

### 一句话记忆

**`nn.Linear` = `x @ W.T + b`，只改最后一维，前面维度全是 batch，自动广播。**

---------

#### ⭐⭐11. Self-Attention vs Cross-Attention 精简对照

------

## 核心区别一句话

> **Self-attention**：Q、K、V 都来自**同一个序列** —— "我在我自己内部做信息聚合" **Cross-attention**：Q 来自**一个序列**，K、V 来自**另一个序列** —— "我用我的问题，去别人那里找答案"

------

## 形象对照

|               | Self-Attention            | Cross-Attention                    |
| ------------- | ------------------------- | ---------------------------------- |
| **Q 来自**    | x                         | x_A（"提问方"）                    |
| **K, V 来自** | x（同一个）               | x_B（"被询问方"）                  |
| **类比**      | 班里同学互相讨论          | 学生向图书馆查资料                 |
| **典型用途**  | 让序列内部 token 互通信息 | 让一个序列**关注另一个序列**的内容 |

------

## 代码上的唯一区别

```python
# Self-attention
q = query(x)           # 同一个 x
k = key(x)
v = value(x)

# Cross-attention
q = query(x_decoder)   # Q 来自一边
k = key(x_encoder)     # K, V 来自另一边
v = value(x_encoder)
```

**就这一个改动**。`q @ k.T`、softmax、`wei @ v` 这些后续操作**一模一样**。

------

## Karpathy 提到的 Transformer 用到 cross-attention 是什么意思？

经典 Transformer (Vaswani 2017) 是 **encoder-decoder** 架构，用于机器翻译：

```
英文输入 → Encoder (self-attention) → encoder 输出
                                          │
                                          ↓
法文译文 → Decoder ──→ self-attention ──→ cross-attention ──→ 输出下一个法文词
                       (看自己已生成的)    (Q=译文, K/V=英文)
```

Decoder 里**两种 attention 都用**：

1. **Self-attention**：让译文 token 看自己之前生成的内容（带 causal mask）
2. **Cross-attention**：让译文 token 去"查"英文原文 —— Q 是当前正在翻译的位置，K/V 是整个英文句子

这就是为什么翻译模型能"对齐"——译文的每个词通过 cross-attention 找到它对应的英文词。

------

## 当下你在学的 nanoGPT 用的是哪种？

**纯 self-attention**（decoder-only 架构）。

GPT 系列、LLaMA、Claude 这些 LLM 都是 **decoder-only**：

- 没有 encoder
- 没有 cross-attention
- 只有带 causal mask 的 self-attention 反复堆叠

**为什么不需要 cross-attention？** 因为对 LLM 来说，"输入"和"输出"都在同一个 token 序列里（prompt + 续写），用 self-attention 就够了。这是 2018 之后 LLM 演化的关键简化。

> 所以你学完 nanoGPT，自然就懂了现代 LLM 的核心。Cross-attention 主要在翻译、image captioning、Stable Diffusion（文本→图像）这类**多模态/跨序列**任务里才出现。

------

## 一张表收尾

| 场景                                 | 架构                 | Attention 类型                            |
| ------------------------------------ | -------------------- | ----------------------------------------- |
| **GPT / Claude / LLaMA**（你正在学） | decoder-only         | self-attention (causal)                   |
| **BERT**                             | encoder-only         | self-attention (无 mask)                  |
| **原版 Transformer / T5**（翻译）    | encoder-decoder      | encoder: self / decoder: self + **cross** |
| **Stable Diffusion**                 | U-Net + text encoder | **cross**（图像查文本）                   |

------

## 一句话总结

> **Self-attention：Q/K/V 同源 → 序列内部通信；Cross-attention：Q 一边、K/V 另一边 → 两个序列之间通信。** 区别就一行代码：`q = query(x_A); k = key(x_B); v = value(x_B)`。你现在学的 GPT 系列全是纯 self-attention，cross-attention 只在 encoder-decoder 或多模态架构里才登场。

-------

#### ⭐⭐12. Scaled Dot-Product Attention 的因果链 —— 为什么除以 √d_k

整条链的核心矛盾是：**"点积维度越高、方差越大；方差越大、softmax 越尖锐；越尖锐、梯度越死"**。

让我从源头一步步推到底。

------

## 第 1 环：点积的方差天然随维度增长

假设 q 和 k 是 `d_k` 维向量，每个元素是均值 0、方差 1 的独立随机变量（初始化时大致如此）：

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

利用独立性：

- $\mathbb{E}[q_i k_i] = 0$
- $\text{Var}(q_i k_i) = 1$
- 求和 d_k 项独立变量 → **总方差 = d_k**

> **点积的标准差 = √d_k**。维度越高，点积的"波动范围"越大。

举个数字感受一下：

- d_k = 4 → 点积典型大小 ≈ ±2
- d_k = 64 → 点积典型大小 ≈ ±8
- d_k = 512 → 点积典型大小 ≈ ±22

------

## 第 2 环：softmax 对"输入大小"极其敏感

softmax 不是缩放等价的（不像加常数那样无害）：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**指数函数把差距放大**。看一个直观对比：

| 输入           | softmax 输出                                      |
| -------------- | ------------------------------------------------- |
| `[1, 2, 3]`    | `[0.09, 0.24, 0.67]` — 较温和分布                 |
| `[10, 20, 30]` | `[≈2e-9, ≈4.5e-5, ≈1.0]` — **几乎全押到最大那个** |

输入只是放大 10 倍，输出就从"软关注几个候选"变成"硬选一个"。

------

## 第 3 环：方差大的点积 + softmax = 注意力过度集中

把第 1、2 环连起来：

- d_k 大 → 点积值波动范围大（典型 ±√d_k）
- 这些大值直接喂给 softmax → softmax 极度尖锐 → **几乎变成 one-hot**

**后果**：

- 每个 query 只关注**一两个 token**，其他全被忽略
- 失去了 attention 应有的"分布式信息聚合"能力
- 多头机制的"分工"也失去意义——每个头都只盯一两个 token

------

## 第 4 环：尖锐 softmax = 梯度近乎消失

softmax 的雅可比矩阵元素是：

$$\frac{\partial \text{softmax}(x)_i}{\partial x_j} = \text{softmax}(x)*i \cdot (\delta*{ij} - \text{softmax}(x)_j)$$

当 softmax 接近 one-hot 时（比如某项 ≈ 1，其余 ≈ 0）：

- 大部分位置的导数 ≈ **0**（因为 `softmax(x)_i ≈ 0`）
- 即使对最大的那项，导数也 ≈ `1 × (1 - 1) = 0`

**整个 softmax 几乎不传梯度**。模型一旦掉进这种状态，attention 权重就训不动了。

> 这和 sigmoid 在 |x| 很大时梯度消失是同一类问题，叫做 **saturation（饱和）**。

------

## 第 5 环：除以 √d_k 一招治本

既然方差是 d_k、标准差是 √d_k，那只要让点积**除以 √d_k**，就能把方差拉回到 1：

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

这样不管 d_k 是 16 还是 512，点积的典型大小都稳定在 **±1** 左右，softmax 始终工作在"温和、可微分"的区间。

代码就一行：

```python
wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)
```

------

## 完整因果链（一图收尾）

```
d_k 增大
   │
   ↓ （概率论）
点积方差 = d_k，标准差 = √d_k
   │
   ↓ （softmax 对幅值敏感）
softmax 输出变尖锐 → 趋近 one-hot
   │
   ├──→ 注意力过度集中：每个 token 只看一两个邻居
   │
   └──→ softmax 梯度近乎为 0 → 训练停滞

—— 解决方案 ——

点积 ÷ √d_k → 方差恢复到 1 → softmax 平滑可微
   │
   └──→ attention 多样、可学习、梯度健康
```

------

## 几个值得知道的延伸点

### 1. 为什么是 √d_k 而不是 d_k？

**控的是标准差**，不是方差。softmax 关心的是输入幅值的尺度（线性的），不是幅值的平方。除以 √d_k 才能让幅值稳定到 O(1)，除以 d_k 会过度抑制。

### 2. 训练初期更关键

初始化时 q、k 的方差就是 1（Xavier/Kaiming 初始化的设计），所以理论方差 = d_k 是**严格成立**的。这一步在训练刚启动时最关键——如果一开始 softmax 就饱和，模型根本学不动。训练几步之后 q、k 的实际分布会偏离假设，但缩放因子仍然是个稳定的"安全垫"。

### 3. 这就是 Transformer 论文标题里的 "Scaled"

原论文叫 **Scaled** Dot-Product Attention，强调的就是这个 √d_k。后来 Vaswani 在论文里专门加了脚注解释这个 scale 的来源——就是上面的方差推导。

### 4. 还有一种缓解方案叫"温度"

更一般地，softmax 可以写成 `softmax(x / T)`，T 就是"温度"：

- T 大 → 输出平滑
- T 小 → 输出尖锐

`√d_k` 本质上就是一个**根据维度自动选定的温度参数**。生成时调 `temperature` 参数（你后面 generate 阶段会遇到）也是同一个原理。

------

## 一句话总结

> 高维点积天然方差大 → softmax 被推到饱和区 → 注意力变 one-hot 且梯度死亡。除以 √d_k 把方差拉回 1 → softmax 始终温和可微 → attention 多样、训练稳定。这是 **Scaled Dot-Product Attention** 名字里 "Scaled" 的全部由来。

----------

#### 13.  `register_buffer`

**官方名称：** `register_buffer`
**官方定义：** 将一个张量注册为模块的缓冲区（buffer），使其成为模块状态的一部分，但**不是可训练参数**。

**Signature：**
```python
module.register_buffer(name, tensor, persistent=True)
```

**一句话核心：** 注册一个**不需要梯度**的张量，但它会随着模型 `.to(device)` 一起移动，并随 `.state_dict()` 保存/加载。

---

### 与 `Parameter` 的对比

| 特性                    | `nn.Parameter` | `register_buffer`      |
| ----------------------- | -------------- | ---------------------- |
| 是否可训练              | ✅ 是           | ❌ 否                   |
| 是否被优化器更新        | ✅ 是           | ❌ 否                   |
| `requires_grad`         | `True`         | `False`                |
| 随 `.to(device)` 移动   | ✅              | ✅                      |
| 随 `.state_dict()` 保存 | ✅              | ✅（`persistent=True`） |

---

### 示例

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 可训练参数
        self.weight = nn.Parameter(torch.randn(10, 10))
        
        # 缓冲区（会保存，会移动）
        self.register_buffer("running_mean", torch.zeros(10))
        self.register_buffer("running_var", torch.ones(10))
        
        # 临时缓冲区（不保存）
        self.register_buffer("temp", torch.randn(5), persistent=False)

model = MyModule()
model.to('cuda')  # weight, running_mean, running_var, temp 都移到 GPU
```

---

### 典型应用场景

| 场景                 | 示例                             |
| -------------------- | -------------------------------- |
| **BatchNorm 统计量** | `running_mean`, `running_var`    |
| **位置编码**         | Transformer 的固定正弦位置编码   |
| **掩码**             | 因果掩码、padding 掩码           |
| **预计算常量**       | 注意力中的缩放因子 `1/sqrt(d_k)` |

---

### 访问缓冲区

```python
# 方式1：直接属性访问
model.running_mean

# 方式2：通过 get_buffer
model.get_buffer('running_mean')

# 方式3：列出所有缓冲区
for name, buf in model.named_buffers():
    print(name, buf.shape)
```

---

### `persistent` 参数

| `persistent`   | `.state_dict()` 包含 | 用途                           |
| -------------- | -------------------- | ------------------------------ |
| `True`（默认） | ✅                    | 需要保存的，如 running_mean    |
| `False`        | ❌                    | 临时缓冲区，如推理时的中间结果 |

```python
# 不保存到 state_dict（每次重新计算）
self.register_buffer("cache", torch.zeros(100), persistent=False)
```

---

### 注意事项

1. **缓冲区默认不需要梯度**（`requires_grad=False`）
2. **会随模型移动**：`.to(device)` 自动处理
3. **可被 `model.state_dict()` 保存**（除非 `persistent=False`）
4. **优化器不会更新它们**

---

### 一句话记忆

**`register_buffer` = 注册非训练张量，会搬家会保存，常用于 BN 统计量、固定掩码、位置编码。**

------

#### 14. 可学习位置编码的模型的 context window

 LLM 最关键的结构限制之一。让我再帮你把这个理解钉牢，并指出几个值得知道的延伸。

------

## 你说的完全对

只要模型用的是 **可学习的位置编码**（`nn.Embedding(block_size, n_embd)` 这种形式），就**结构性地**只能处理最多 `block_size` 个 token。每次 forward 时，能用作上下文的最多就是**最近 `block_size` 个 token**——更早的只能丢弃。

这就是为什么所有 LLM 都有"context window"这个术语：

| 模型              | context window | 含义                     |
| ----------------- | -------------- | ------------------------ |
| GPT-2             | 1024           | 一次最多看 1024 个 token |
| GPT-3             | 2048           | 一次最多看 2048 个 token |
| GPT-4 (经典版)    | 8192           | 一次最多看 8192 个 token |
| Claude 3.5 Sonnet | 200K           | 一次最多看 200K 个 token |
| 你的 nanoGPT      | 8 (block_size) | 一次最多看 8 个 token    |

**"context window" 这个词，本质就是 "position embedding 表的行数"。**

------

## 但要注意一个微妙之处：不是"截断"等于"看不见"

虽然每次 forward 只能看最近 `block_size` 个 token，**但模型在生成过程中是通过 KV cache（或重复 forward）逐步推进的**——更早的信息其实**已经融入了被截断后留下的那些 token 的内部表示**（如果模型有多层 attention 的话）。

不过……对于你现在的玩具 nanoGPT 来说，generate 函数是 **stateless** 的（每次 forward 都从头算），所以"截掉的 token 完全消失"这个理解是准确的。等你以后接触 KV cache 实现，这件事会稍微复杂一点，但本质约束没变。

------

## 这条 hard constraint 带来了哪些深远影响？

### 1. **为什么"长文档总结"这么难**

如果你给 Claude 喂 100 万字的小说，光是把它放进 context window 就是巨大挑战。各家公司拼 context length（4K → 32K → 200K → 1M）就是在和这个 hard constraint 做斗争。

### 2. **为什么需要 RAG（检索增强）**

context window 装不下整个知识库，只能用检索把"最相关的片段"塞进 window 里。

### 3. **为什么有 RoPE / ALiBi 这类位置编码**

它们**不查表，而是用数学公式直接算位置信息**——理论上没有 block_size 上限，可以"外推"到训练时没见过的长度。这是 LLaMA、Mistral 这些现代模型抛弃 `nn.Embedding(block_size, n_embd)` 的核心原因。

### 4. **为什么有 sliding window attention（Mistral、Longformer）**

即使 position 表能扩，attention 矩阵的内存是 O(T²)。所以现代模型常常用"滑动窗口 attention"——每个 token 只看最近 W 个邻居，让 T 不受 attention 内存约束。

### 5. **为什么 ChatGPT 长对话会"忘事"**

对话太长超过 context window，早期消息会被丢掉或压缩。

------

## 一个稍微更精确的说法

你的总结 ✓ 但可以再精炼一层：

> **可学习位置编码的模型**有一个天生的 hard constraint —— **每次 forward 最多只能处理 block_size 个 token**。所以在 generate 这种自回归循环里，**喂进 forward 的 idx 必须截断到最近 block_size 个 token**。

注意我加的限定："可学习位置编码"。这把约束的来源说清楚了——不是 attention 本身的限制，也不是 transformer 架构的限制，而是**这种特定位置编码方案的限制**。RoPE / ALiBi 等方案就没有这个 hard constraint。

------

## 一句话总结

> **正确。** 用 `nn.Embedding(block_size, n_embd)` 这种可学习位置编码的模型，结构上写死了"最多看 block_size 个 token"。generate 时的截断 `idx[:, -block_size:]` 不是 workaround，而是与这个 hard constraint 共处的**标准做法**。所有 LLM 的 "context window" 概念，本质就是这条 hard constraint 的别名。

----------

#### 15. `nn.Sequential`

**官方名称：** `torch.nn.Sequential`
**官方定义：** 一个顺序容器，其中包含的模块将按传入构造器的顺序被依次调用。

**Signature：**

```python
nn.Sequential(*args)
# args: 多个 nn.Module 实例或 OrderedDict
```

**一句话核心：** 将多个层**串起来**，数据依次流过每一层，前一个输出自动作为后一个输入。

----

#### 16. LayerNorm

## 和BatchNorm的核心区别：归一化的"方向"不同

假设有一个 `(B, T, C)` 的张量（B 个样本，每个长度 T，每个 token 有 C 个特征）。

|               | 沿哪个维度算 mean/std                              | 每个统计量被多少元素共享                       |
| :------------ | :------------------------------------------------- | :--------------------------------------------- |
| **BatchNorm** | 沿 **B 维**（同一 batch 内所有样本的同一特征），行 | 一个特征通道一套统计量，被 B×T 个元素共享      |
| **LayerNorm** | 沿 **C 维**（一个 token 自己的所有特征），列       | 每个 (b, t) 位置自己一套统计量，独立于其他位置 |

### 一句话直觉

- **BatchNorm**：*"看一眼整个 batch，把每个特征通道标准化"*
- **LayerNorm**：*"我（一个 token）自己内部把我所有特征标准化"*

Transformer 里 token 是个 C 维向量，承载该 token 的语义。**沿 C 维归一化 = 把这个 token 的"语义向量"标准化**，符合"每个 token 单位的处理"这一架构哲学。BatchNorm 沿 B 维统计，等于"用一堆不相关的 token 来归一化我"，语义上不太对。

---------

#### 17. Pre-LN vs Post-LN

## 两种写法

```python
# Post-LN（原 Transformer 论文 2017）
x = LayerNorm(x + attention(x))
x = LayerNorm(x + ffn(x))

# Pre-LN（GPT-2 之后的现代标配）
x = x + attention(LayerNorm(x))
x = x + ffn(LayerNorm(x))
```

差别：**LayerNorm 放在子层之前（Pre）还是之后（Post）**。

## 核心区别：残差通路是否"干净"

- **Pre-LN**：残差主路是**干净的恒等映射**，LayerNorm 只在支路上 → 梯度可无损直通深层
- **Post-LN**：主路被 LayerNorm 反复缩放 → 深层网络梯度爆炸/消失

## Pre-LN 的优势

| 维度           | Post-LN                 | Pre-LN          |
| -------------- | ----------------------- | --------------- |
| 训练稳定性     | 必须用 warmup 才不发散  | 无需 warmup     |
| 支持的网络深度 | 12 层以上就难训         | 几十~上百层都稳 |
| 超参敏感度     | 高（lr、init 都要细调） | 低              |
| 最终精度       | 训得起来时略高          | 略低但可忽略    |

> 实证依据：Xiong et al. 2020 *"On Layer Normalization in the Transformer Architecture"* 证明 Post-LN 梯度随深度指数增长，Pre-LN 与深度无关。

## Pre-LN 的小代价

- 残差累积流越加越大 → 通常在网络出口处再加一个 LayerNorm 兜底：

  ```python
  for block in self.blocks:    x = block(x)x = self.ln_f(x)   # 最终 LayerNorm
  ```

## 结论

**GPT-2 之后所有大模型（GPT-3/4、Claude、LLaMA、PaLM、Mistral）全部用 Pre-LN**，这是大模型能堆深的关键工程突破之一。

## 直觉记忆

> Pre-LN = 主干道全程畅通，支路上才有红绿灯； Post-LN = 每个路口都有红绿灯，深一点就堵车（梯度死）。