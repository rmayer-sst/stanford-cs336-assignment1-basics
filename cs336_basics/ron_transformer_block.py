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

class TransformerBlock(nn.Module):
    """

        Implement the pre-norm Transformer block as described in §3.5 and illustrated in Figure 2. Your
        Transformer block should accept (at least) the following parameters.

        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.

        To test your implementation, implement the adapter [adapters.run_transformer_block]. Then
        run uv run pytest -k test_transformer_block to test your implementation.
        Deliverable: Transformer block code that passes the provided tests.



    Let’s begin by assembling the Transformer block (it will be helpful to refer
    back to Figure 2). A Transformer block contains two ‘sublayers’, one for the
    multihead self attention, and another for the feed-forward network. In each
    sublayer, we first perform RMSNorm, then the main operation (MHA/FF), finally
    adding in the residual connection. To be concrete, the first half (the first
    ‘sub-layer’) of the Transformer block should be implementing the following set
    of updates to produce an output y from an input x,
        y = x + MultiHeadSelfAttention(RMSNorm(x)). (15)

    Given the weights of a pre-norm Transformer block and input features, return
    the output of running the Transformer block on the input features.

    This function should use RoPE. Depending on your implementation, you may
    simply need to pass the relevant args to your TransformerBlock constructor, or
    you may need to initialize your own RoPE class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention.
            `d_model` must be evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your
            implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model). The rows are ordered by matrices of
                shape (num_heads, d_k), so
                `attn.q_proj.weight == torch.cat([q_heads.0.weight, ...,
                q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model). The rows are ordered by matrices of
                shape (num_heads, d_k), so
                `attn.k_proj.weight == torch.cat([k_heads.0.weight, ...,
                k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model). The rows are ordered by matrices of
                shape (num_heads, d_v), so
                `attn.v_proj.weight == torch.cat([v_heads.0.weight, ...,
                v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection.
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm applied in the
                transformer block. Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm applied in the
                transformer block. Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float,
                 ):
        super().__init__()
        self.cmsawr = CausalMultiheadSelfAttentionWithRope(d_model, num_heads, theta, max_seq_len)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)


    def forward(self, x: Float[Tensor, "batch sequence_length d_model"], return_attention: bool = False):
        """
            To be concrete, the first half (the first ‘sub-layer’) of the Transformer 
            block should be implementing the following set of updates to 
            produce an output y from an input x:
            y = x + MultiHeadSelfAttention(RMSNorm(x)). (15)
        """
        attn_out, attn_weights = self.cmsawr.forward(self.ln1(x), torch.arange(x.shape[1]), return_attention=True)
        y = x + attn_out
        """
            Note -- in MoE architectures, only the following expression would change.
            https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/master/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf
            "What’s a MoE?
             Replace big feedforward with (many) big feedforward networks and a selector layer"
            That's the feedforward network they're talking about.
        """
        z = self.ffn(self.ln2(y))
        output = y + z
        if return_attention:
            return output, attn_weights
        return output