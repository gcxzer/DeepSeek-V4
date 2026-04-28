import importlib
import math
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F


# Debug examples:
#   uv run python -m unittest tests.test_model -v
#   uv run python -m unittest tests.test_model.CompressorTests -v
#   uv run python -m unittest tests.test_model.CompressorTests.test_decode_compresses_only_when_ratio_boundary_is_reached -v

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = ROOT / "Inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))


def _install_cpu_kernel_stub():
    """Let model.py import without tilelang/GPU kernels.

    These tests focus on Python shape/control-flow logic. Quantization kernels are
    treated as in-place no-ops so Compressor can run on CPU.
    """
    kernel = types.ModuleType("kernel")

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        if inplace:
            return x
        scale_shape = (*x.shape[:-1], math.ceil(x.shape[-1] / block_size))
        return x, torch.ones(scale_shape, dtype=scale_dtype, device=x.device)

    def fp4_act_quant(x, block_size=32, inplace=False):
        if inplace:
            return x
        scale_shape = (*x.shape[:-1], math.ceil(x.shape[-1] / block_size))
        return x, torch.ones(scale_shape, dtype=torch.float32, device=x.device)

    def fp8_gemm(x, x_scale, weight, weight_scale, scale_dtype):
        return F.linear(x.float(), weight.float()).to(x.dtype)

    def fp4_gemm(x, x_scale, weight, weight_scale, scale_dtype):
        return F.linear(x.float(), weight.float()).to(x.dtype)

    def sparse_attn(*args, **kwargs):
        raise NotImplementedError("sparse_attn is not needed by these unit tests")

    def hc_split_sinkhorn(*args, **kwargs):
        raise NotImplementedError("hc_split_sinkhorn is not needed by these unit tests")

    kernel.act_quant = act_quant
    kernel.fp4_act_quant = fp4_act_quant
    kernel.fp8_gemm = fp8_gemm
    kernel.fp4_gemm = fp4_gemm
    kernel.sparse_attn = sparse_attn
    kernel.hc_split_sinkhorn = hc_split_sinkhorn
    sys.modules["kernel"] = kernel


_install_cpu_kernel_stub()
model = importlib.import_module("model")


@contextmanager
def model_world_size(value):
    previous = model.world_size
    model.world_size = value
    try:
        yield
    finally:
        model.world_size = previous


def small_args(**overrides):
    values = dict(
        max_batch_size=2,
        max_seq_len=16,
        dim=6,
        head_dim=8,
        rope_head_dim=4,
        norm_eps=1e-6,
        compress_ratios=(0,),
        dtype="bf16",
        scale_fmt=None,
        scale_dtype="fp32",
    )
    values.update(overrides)
    return model.ModelArgs(**values)


def init_compressor_for_debug(compressor):
    """Use deterministic parameters so failures are easier to inspect."""
    with torch.no_grad():
        compressor.ape.zero_()
        compressor.norm.weight.fill_(1.0)
        compressor.wkv.weight.zero_()
        compressor.wgate.weight.zero_()

        # Make wkv read the first head_dim input channels, wrapping if dim < head_dim.
        for out_idx in range(compressor.wkv.weight.shape[0]):
            in_idx = out_idx % compressor.dim
            compressor.wkv.weight[out_idx, in_idx] = 1.0

        # Zero gate scores mean softmax becomes an average within each group.
        # This makes the expected pooling behavior easy to reason about.
        if compressor.wkv.bias is not None:
            compressor.wkv.bias.zero_()
        if compressor.wgate.bias is not None:
            compressor.wgate.bias.zero_()


class DTypeTests(unittest.TestCase):
    def test_set_dtype_restores_default_after_success_and_exception(self):
        original = torch.get_default_dtype()

        with model.set_dtype(torch.float64):
            self.assertEqual(torch.get_default_dtype(), torch.float64)

        self.assertEqual(torch.get_default_dtype(), original)

        with self.assertRaises(RuntimeError):
            with model.set_dtype(torch.float64):
                raise RuntimeError("debug exception")

        self.assertEqual(torch.get_default_dtype(), original)


class ParallelLinearTests(unittest.TestCase):
    def test_column_parallel_linear_splits_output_features(self):
        with model_world_size(2):
            layer = model.ColumnParallelLinear(3, 8, dtype=torch.float32)

        self.assertEqual(layer.part_out_features, 4)
        self.assertEqual(tuple(layer.weight.shape), (4, 3))

        with torch.no_grad():
            layer.weight.copy_(torch.arange(12, dtype=torch.float32).view(4, 3))

        x = torch.arange(15, dtype=torch.float32).view(5, 3)
        y = layer(x)

        self.assertEqual(tuple(y.shape), (5, 4))
        torch.testing.assert_close(y, F.linear(x, layer.weight))

    def test_row_parallel_linear_splits_input_features_and_adds_bias_after_reduce(self):
        with model_world_size(2):
            layer = model.RowParallelLinear(6, 4, bias=True, dtype=torch.float32)

        self.assertEqual(layer.part_in_features, 3)
        self.assertEqual(tuple(layer.weight.shape), (4, 3))
        self.assertEqual(tuple(layer.bias.shape), (4,))

        with torch.no_grad():
            layer.weight.copy_(torch.arange(12, dtype=torch.float32).view(4, 3) / 10)
            layer.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

        x = torch.arange(15, dtype=torch.float32).view(5, 3)
        expected_partial = F.linear(x, layer.weight)

        with model_world_size(2), patch.object(model.dist, "all_reduce", side_effect=lambda tensor: tensor):
            y = layer(x)

        self.assertEqual(tuple(y.shape), (5, 4))
        torch.testing.assert_close(y, expected_partial + layer.bias)


class CompressorTests(unittest.TestCase):
    def make_compressor(self, compress_ratio=2, rotate=False):
        args = small_args()
        compressor = model.Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim, rotate=rotate)
        init_compressor_for_debug(compressor)
        compressor.kv_cache = torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, args.head_dim)
        compressor.freqs_cis = model.precompute_freqs_cis(
            args.rope_head_dim,
            args.max_seq_len,
            args.original_seq_len,
            args.rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        return compressor

    def test_overlap_transform_moves_previous_window_into_first_half(self):
        compressor = self.make_compressor(compress_ratio=4)
        tensor = torch.arange(1 * 3 * 4 * 16, dtype=torch.float32).view(1, 3, 4, 16)

        transformed = compressor.overlap_transform(tensor, value=-1)

        self.assertEqual(tuple(transformed.shape), (1, 3, 8, 8))
        torch.testing.assert_close(transformed[:, :, 4:], tensor[:, :, :, 8:])
        torch.testing.assert_close(transformed[:, 1:, :4], tensor[:, :-1, :, :8])
        self.assertTrue((transformed[:, 0, :4] == -1).all())

    def test_prefill_compresses_full_groups_and_stores_remainder_state(self):
        compressor = self.make_compressor(compress_ratio=2)
        x = torch.arange(1 * 5 * 6, dtype=torch.float32).view(1, 5, 6)

        kv = compressor(x, start_pos=0)

        self.assertIsNotNone(kv)
        self.assertEqual(tuple(kv.shape), (1, 2, 8))
        self.assertFalse(torch.isnan(kv).any())
        torch.testing.assert_close(compressor.kv_cache[:1, :2], kv)

        # The fifth token is not enough to form a group, so it waits in state.
        raw_remainder = compressor.wkv(x.float())[:, 4]
        torch.testing.assert_close(compressor.kv_state[:1, 0], raw_remainder)

    def test_prefill_returns_none_when_sequence_shorter_than_ratio(self):
        compressor = self.make_compressor(compress_ratio=4)
        x = torch.arange(1 * 3 * 6, dtype=torch.float32).view(1, 3, 6)

        kv = compressor(x, start_pos=0)

        self.assertIsNone(kv)
        self.assertTrue((compressor.kv_cache == 0).all())
        torch.testing.assert_close(compressor.kv_state[:1, 4:7], compressor.wkv(x.float()))

    def test_decode_compresses_only_when_ratio_boundary_is_reached(self):
        compressor = self.make_compressor(compress_ratio=2)

        first = torch.arange(1 * 1 * 6, dtype=torch.float32).view(1, 1, 6)
        second = first + 10

        self.assertIsNone(compressor(first, start_pos=0))
        kv = compressor(second, start_pos=1)

        self.assertIsNotNone(kv)
        self.assertEqual(tuple(kv.shape), (1, 1, 8))
        torch.testing.assert_close(compressor.kv_cache[:1, 0], kv.squeeze(1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
