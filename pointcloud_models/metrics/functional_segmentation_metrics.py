import warnings

import torch
import typing as t

__all__ = ("iou_score", "accuracy")

# this implementation is taken from Pytorch Segmentation Models - thanks for the nice implementation!
# https://github.com/qubvel/segmentation_models.pytorch


def _iou_score(tp, fp, fn, tn):
    return tp / (tp + fp + fn)


def _accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)


def _handle_zero_division(x, zero_division):
    nans = torch.isnan(x)
    if torch.any(nans) and zero_division == "warn":
        warnings.warn("Zero division in metric calculation!")
    value = zero_division if zero_division != "warn" else 0
    value = torch.tensor(value, dtype=x.dtype).to(x.device)
    x = torch.where(nans, value, x)
    return x


def _compute_metric(
        metric_fn,
        tp,
        fp,
        fn,
        tn,
        reduction: t.Optional[str] = None,
        zero_division="warn",
        **metric_kwargs,
) -> t.Union[float, torch.Tensor]:
    if reduction == "micro":
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)

    elif reduction == "macro":
        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = score.mean()

    elif reduction == "none" or reduction is None:
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)

    else:
        raise ValueError(
            "`reduction` should be in [micro, macro, none, None]"
        )

    return score


def iou_score(
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
        tn: torch.Tensor,
        reduction: t.Optional[str] = None,
        zero_division: t.Union[str, float] = 1.0,
) -> torch.Tensor:
    """IoU score or Jaccard index"""  # noqa
    return _compute_metric(
        _iou_score,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        zero_division=zero_division,
    )


def accuracy(
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
        tn: torch.Tensor,
        reduction: t.Optional[str] = None,
        zero_division: t.Union[str, float] = 1.0,
) -> torch.Tensor:
    """Accuracy"""
    return _compute_metric(
        _accuracy,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        zero_division=zero_division,
    )
