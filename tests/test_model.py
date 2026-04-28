import torch
from contextlib import contextmanager

@contextmanager
def set_dtype(dtype):
    """Temporarily override torch default dtype, restoring it on exit (even if an exception occurs)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


if __name__ == "__main__":
    print("before:", torch.get_default_dtype())  # 通常是 float32

    with set_dtype(torch.float64):
        x = torch.tensor([1.0, 2.0])   # 按 float64 创建
        w = torch.randn(2, 2)          # 也会是 float64
        print("inside:", x.dtype, w.dtype)

    print("after:", torch.get_default_dtype())  # 恢复为原值
