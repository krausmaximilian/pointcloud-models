import torch
import typing as t
from torch import Tensor

__all__ = ("get_segmentation_statistics",)


@torch.no_grad()
def get_segmentation_statistics(output: t.Union[torch.LongTensor, torch.FloatTensor], labels: torch.LongTensor,
                                num_classes: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if torch.is_floating_point(labels):
        raise ValueError(f"Labels should have int types, got {labels.dtype}.")

    if output.shape != labels.shape:
        raise ValueError(
            "Dimensions should match, but ``output`` shape is not equal to ``labels`` "
            + f"shape, {output.shape} != {labels.shape}"
        )
    batch_size, *dims = output.shape
    num_elements = torch.prod(torch.tensor(dims)).long()

    tps = torch.zeros(batch_size, num_classes, dtype=torch.long)
    fps = torch.zeros(batch_size, num_classes, dtype=torch.long)
    fns = torch.zeros(batch_size, num_classes, dtype=torch.long)
    tns = torch.zeros(batch_size, num_classes, dtype=torch.long)

    for i in range(batch_size):
        target_i = labels[i]
        output_i = output[i]
        mask = output_i == target_i
        matched = torch.where(mask, target_i, -1)
        tp = torch.histc(matched.float(), bins=num_classes, min=0, max=num_classes - 1)
        fp = torch.histc(output_i.float(), bins=num_classes, min=0, max=num_classes - 1) - tp
        fn = torch.histc(target_i.float(), bins=num_classes, min=0, max=num_classes - 1) - tp
        tn = num_elements - tp - fp - fn
        tps[i] = tp.long()
        fps[i] = fp.long()
        fns[i] = fn.long()
        tns[i] = tn.long()

    return tps, fps, fns, tns
