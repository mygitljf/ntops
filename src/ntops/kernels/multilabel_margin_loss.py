import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


BLOCK_SIZE = 128


def loss_arrangement(input, target, output, class_block=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.tile((1, class_block)).squeeze(1)
    target_arranged = target.tile((1, class_block)).squeeze(1)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    target_arranged.dtype = target_arranged.dtype.squeeze(0)
    output_arranged = output.tile((1,))
    return input_arranged, target_arranged, output_arranged


def loss_application(input, target, output):
    class_id = input.offsets(-1)
    valid_class = class_id < input.source.shape[-1]
    input_f32 = ntl.cast(input, ntl.float32)
    target_values = target

    slot = input.offsets(-1)
    end_of_targets = input.source.shape[-1] + 1
    first_negative = ntl.min(ntl.where(target_values < 0, slot, end_of_targets), axis=0)
    active_slot = slot < first_negative

    target_match = (target_values[:, None] == class_id[None, :]) & active_slot[:, None]
    target_count = ntl.sum(ntl.where(target_match, 1.0, 0.0), axis=0)
    is_non_target = valid_class & (target_count == 0)

    margin = 1.0 - input_f32[:, None] + input_f32[None, :]
    hinge = ntl.where(margin > 0.0, margin, 0.0)
    weight = target_count[:, None] * ntl.where(is_non_target[None, :], 1.0, 0.0)
    total = ntl.sum(ntl.sum(weight * hinge, axis=0), axis=0)

    output = total / input.source.shape[-1]  # noqa: F841


def premake_loss(batch, num_classes, class_block, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(
        loss_arrangement,
        class_block=class_block,
        block_size=block_size,
    )
    tensors = (
        Tensor(2, shape=(batch, num_classes), dtype=dtype, other=0),
        Tensor(2, shape=(batch, num_classes), dtype=None, other=-1),
        Tensor(1, shape=(batch,), dtype=dtype),
    )
    return arrangement_, loss_application, tensors
