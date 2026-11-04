"""Grouped attention: native CUDA GQA, explicit CPU head repetition for tests."""

from torch.nn import functional


def sdpa(query, key, value, attn_mask):
    if query.device.type == "cpu":
        repeat = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        return functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask
        )
    return functional.scaled_dot_product_attention(  # type: ignore[call-arg]
        query, key, value, attn_mask=attn_mask, enable_gqa=True
    )
