# Tensor Parallel Linear Notes

这份笔记用来理解下面两个类的区别和先后用法：

- `ColumnParallelLinear`: 按输出维度切分权重和输出。
- `RowParallelLinear`: 按输入维度切分权重，最后对输出做 `all_reduce` 汇总。

![ColumnParallelLinear and RowParallelLinear](assets/tensor-parallel-linear.png)

## 一句话理解

`ColumnParallelLinear` 负责把一个 Linear 的输出维度拆到多个 TP rank 上；每个 rank 只算自己那一段输出，所以输出本身就是分片的，不需要通信。

`RowParallelLinear` 负责消费已经按输入维度分片的张量；每个 rank 算出一个 partial output，然后通过 `dist.all_reduce` 把所有 partial output 相加，得到完整输出。

## ColumnParallelLinear

```python
class ColumnParallelLinear(Linear):
    """Shards output dim across TP ranks. No all-reduce needed on output."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)
```

假设输入是：

```text
x: [seq_len, in_features]
```

原本普通 Linear 的权重逻辑上是：

```text
W: [out_features, in_features]
```

在 `ColumnParallelLinear` 里，`out_features` 被按 `world_size` 切开。每个 rank 只保存：

```text
W_i: [out_features / world_size, in_features]
```

所以每个 rank 的输出是：

```text
y_i: [seq_len, out_features / world_size]
```

这些输出分片逻辑上拼起来才是完整输出：

```text
y = concat(y_0, y_1, ..., y_n)
```

但代码里通常不会马上真的 `concat`，而是让每个 rank 继续拿着自己的分片往后算。

## RowParallelLinear

```python
class RowParallelLinear(Linear):
    """Shards input dim across TP ranks. All-reduce on output to sum partial results."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if world_size > 1:
            y = y.float()
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)
```

`RowParallelLinear` 的输入已经被切成多份：

```text
x_i: [seq_len, in_features / world_size]
```

每个 rank 保存对应的权重分片：

```text
W_i: [out_features, in_features / world_size]
```

每个 rank 先各自算 partial output：

```text
partial_i = x_i @ W_i.T
partial_i: [seq_len, out_features]
```

注意这里每个 rank 算出来的 shape 已经是 `[seq_len, out_features]`，但它只是完整结果的一部分贡献。真正的完整输出需要把所有 rank 的 partial output 相加：

```text
y = partial_0 + partial_1 + ... + partial_n
```

这就是为什么 `RowParallelLinear.forward()` 里需要：

```python
dist.all_reduce(y)
```

`all_reduce` 之后，每个 rank 都拿到相同的完整 `y`。

## 为什么通常先 Column 后 Row

Transformer 里的 MLP 常见结构是：

```text
x -> up/gate projection -> activation -> down projection -> y
```

对应到 tensor parallel，经常是：

```text
x
  -> ColumnParallelLinear
  -> activation
  -> RowParallelLinear
  -> y
```

原因是：

1. `ColumnParallelLinear` 把中间维度切开，每个 rank 得到一段 hidden 分片。
2. activation 可以在每个 rank 本地独立计算，不需要通信。
3. `RowParallelLinear` 正好消费这些输入分片。
4. 最后只在 `RowParallelLinear` 做一次 `all_reduce`，把结果汇总。

## 记忆方式

```text
ColumnParallelLinear:
  split out_features
  output is sharded
  no all-reduce

RowParallelLinear:
  split in_features
  each rank computes partial output
  all-reduce to sum partial outputs
```

更短一点：

```text
Column: 切输出，不汇总
Row: 切输入，要汇总
```
