import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
import math
import einx
from .ron_softmax import softmax

def scaled_dot_product_attention(
        q:Tensor, # (batch_size, ..., seq_len, d_k) # n×dk
        k:Tensor, # (batch_size, ..., seq_len, d_k) # m×dk
        v:Tensor, # (batch_size, ..., seq_len, d_v) # m×dv
        mask: Tensor|None, # (n×m)
        return_attention: bool = False,
):
    """
        Would have been hard if not for: 
        https://einx.readthedocs.io/en/stable/gettingstarted/commonnnops.html
        where they give the harder multi-head case.

        a = einx.dot("b q (h c), b k (h c) -> b q k h", q, k, h=8)
        a = einx.softmax("b q [k] h", a)
        x = einx.dot("b q k h, b k (h c) -> b q (h c)", a, v)

        Replace "b" with "..." because the homework pdf
        said "where ... represents any number of other batch-like dimensions" ...
        and we don't need "h".
    """
    keylen = k.shape[-1]
    a = einx.dot("... q d, ... k d -> ... q k", q, k)
    a /= math.sqrt(keylen)
    if mask is not None:
        a = a.masked_fill(~mask, -float('inf'))
    a = einx.softmax("... q [k]", a) # a = softmax(a,i=-1)
    x = einx.dot("... q k, ... k d -> ... q d", a, v)
    if return_attention:
        return x, a
    return x