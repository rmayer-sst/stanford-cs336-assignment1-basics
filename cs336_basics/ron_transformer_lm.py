import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
import einx
from .ron_rope import RoPE
from .ron_linear import Linear
from .ron_scaled_dot_product_attention import scaled_dot_product_attention
from .ron_rope import RoPE
from .ron_causal_multihead_self_attention_with_rope import CausalMultiheadSelfAttentionWithRope
from .ron_rmsnorm import RMSNorm
from .ron_swiglu import SwiGLU
from .ron_embedding import Embedding
from .ron_transformer_block import TransformerBlock

class TransformerLM(nn.Module):
    """
    Time to put it all together! Implement the Transformer language model as described in §3.1
    and illustrated in Figure 1. At minimum, your implementation should accept all the aforementioned
    construction parameters for the Transformer block, as well as these additional parameters:

        vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token
    embedding matrix.

        context_length: int The maximum context length, necessary for determining the dimensionality of
    the position embedding matrix.

        num_layers: int The number of Transformer blocks to use.

    To test your implementation against our provided tests, you will first need to implement the test
    adapter at [adapters.run_transformer_lm]. Then, run 
       uv run pytest -k test_transformer_lm t -rA --tb=line
    to test your implementation.
    Deliverable: A Transformer LM module that passes the above tests.
    """
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int
                 ):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.xform = nn.ModuleList(
                        TransformerBlock(d_model,num_heads,d_ff,max_seq_len,theta)
                        for _ in range(num_layers)
        )
        self.norm = RMSNorm(d_model)
        self.head = Linear(d_model,vocab_size)

    def forward(self, x, return_attention: bool = False):
        #print("in TransformerLLM x is ",x)
        x = self.embed(x)
        all_attn_weights = []
        for xf in self.xform:
            x, attn_weights = xf(x, return_attention=True)
            all_attn_weights.append(attn_weights)
        x = self.norm(x)
        x = self.head(x)
        if return_attention:
            return x, all_attn_weights
        return x


        



