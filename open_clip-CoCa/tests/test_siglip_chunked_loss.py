"""Parity test for SigLipLoss with chunk_size > 0.

Verifies the chunked path matches the reference (chunk_size=0) bit-exactly
across multiple chunk sizes and both ground-truth modes (negative_only on/off),
in the forward pass and the gradient.
"""
import pytest
import torch

from open_clip.loss import SigLipLoss


def _make_inputs(B=64, D=32, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(B, D, generator=g, dtype=dtype)
    img = img / img.norm(dim=-1, keepdim=True)
    txt = torch.randn(B, D, generator=g, dtype=dtype)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    logit_scale = torch.tensor(10.0, dtype=dtype)
    logit_bias = torch.tensor(-10.0, dtype=dtype)
    return img, txt, logit_scale, logit_bias


@pytest.mark.parametrize("chunk_size", [64, 32, 8, 1])
@pytest.mark.parametrize("negative_only", [False, True])
def test_chunked_matches_reference_forward(chunk_size, negative_only):
    img, txt, ls, lb = _make_inputs(B=64)
    ref = SigLipLoss()._loss(img, txt, ls, lb, negative_only=negative_only)
    chunked = SigLipLoss(chunk_size=chunk_size)._loss(img, txt, ls, lb, negative_only=negative_only)
    assert torch.allclose(ref, chunked, atol=1e-5), \
        f"chunked (cs={chunk_size}, neg_only={negative_only}) differs: ref={ref.item()} chunk={chunked.item()}"


@pytest.mark.parametrize("chunk_size", [32, 8])
def test_chunked_matches_reference_backward(chunk_size):
    img, txt, ls, lb = _make_inputs(B=64)
    img1 = img.clone().requires_grad_(True)
    img2 = img.clone().requires_grad_(True)
    SigLipLoss()._loss(img1, txt, ls, lb).backward()
    SigLipLoss(chunk_size=chunk_size)._loss(img2, txt, ls, lb).backward()
    assert torch.allclose(img1.grad, img2.grad, atol=1e-5)


def test_chunked_handles_uneven_chunks():
    """Last chunk may be smaller than chunk_size; should not error."""
    img, txt, ls, lb = _make_inputs(B=70)  # 70 = 8*8 + 6, so last chunk has 6 rows
    ref = SigLipLoss()._loss(img, txt, ls, lb)
    chunked = SigLipLoss(chunk_size=8)._loss(img, txt, ls, lb)
    assert torch.allclose(ref, chunked, atol=1e-5)


def test_chunk_size_zero_skips_chunked_path():
    """Default behavior (chunk_size=0) must equal the original implementation."""
    img, txt, ls, lb = _make_inputs(B=32)
    ref = SigLipLoss()._loss(img, txt, ls, lb)
    explicit = SigLipLoss(chunk_size=0)._loss(img, txt, ls, lb)
    assert torch.equal(ref, explicit)
