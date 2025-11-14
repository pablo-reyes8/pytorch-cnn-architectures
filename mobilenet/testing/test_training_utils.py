import pytest
import torch

from training.diff_augment import DiffAugment
from training.one_epoch_loop import accuracy_topk


def test_accuracy_topk_matches_manual_computation():
    logits = torch.tensor([[10.0, 0.0, -1.0],
                           [0.0, 5.0, 1.0],
                           [0.2, 0.1, 0.0]])
    targets = torch.tensor([0, 2, 1])

    metrics = accuracy_topk(logits, targets, topk=(1, 3))
    assert metrics["top1"] == pytest.approx(1 / 3)
    assert metrics["top3"] == pytest.approx(1.0)


def test_diffaugment_eval_mode_is_identity():
    augment = DiffAugment(p_flip=1.0,
                          p_brightness=1.0,
                          max_brightness_delta=0.5,
                          p_contrast=1.0,
                          max_contrast_scale=0.5,
                          p_cutout=1.0,
                          cutout_frac=0.5)
    augment.eval()
    x = torch.randn(2, 3, 16, 16)
    y = augment(x.clone())
    assert torch.allclose(y, x)


def test_diffaugment_horizontal_flip_when_training():
    augment = DiffAugment(p_flip=1.0,
                          p_brightness=0.0,
                          p_contrast=0.0,
                          p_cutout=0.0)
    augment.train()
    x = torch.arange(3 * 2 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4)
    y = augment(x.clone())
    expected = torch.flip(x, dims=[3])
    assert torch.allclose(y, expected)
