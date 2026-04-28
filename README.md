# DeepSeek-V4 Notes

这个 README 用来放学习 DeepSeek-V4 代码时整理的图和简短笔记。

## Tensor Parallel Linear

![ColumnParallelLinear and RowParallelLinear](assets/tensor-parallel-linear.png)

关键记忆：

- `ColumnParallelLinear`: 切 `out_features`，每个 rank 得到一段输出，不需要 `all_reduce`。
- `RowParallelLinear`: 切 `in_features`，每个 rank 算 partial output，最后用 `all_reduce` 求和。
- PyTorch `Linear` 的权重形状是 `[out_features, in_features]`，计算时等价于 `x @ weight.T`。
- 在图里的例子中，第一维是 `seq_len`。

常见顺序：

```text
x -> ColumnParallelLinear -> activation -> RowParallelLinear -> y
```
