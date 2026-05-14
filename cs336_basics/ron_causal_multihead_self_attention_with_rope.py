import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
import einx
from .ron_rope import RoPE
from .ron_linear import Linear
from .ron_scaled_dot_product_attention import scaled_dot_product_attention
from .ron_rope import RoPE

# test with:
#  pytest -k with_rope

class CausalMultiheadSelfAttentionWithRope(torch.nn.Module):
    """

    """    
    def __init__(self,
                 d_model: int, 
                 num_heads: int,
                 theta:float,
                 max_seq_len: int
        ):
        super().__init__()
        dk = dv = d_model // num_heads
        self.dk = dk
        self.dv = dv
        self.dm = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, num_heads * dk)
        self.k_proj = Linear(d_model, num_heads * dk)
        self.v_proj = Linear(d_model, num_heads * dv)
        self.o_proj = Linear(num_heads * dv, d_model)
        self.rope = RoPE(d_k = dk, max_seq_len=max_seq_len, theta=theta)

    def make_a_triangle_mask(self,sequence_length:int, device):
        idx = torch.arange(sequence_length,device=device)
        triangle_mask = idx[:, None] >= idx[None, :]     # [L, L], boolean
        return triangle_mask

    def forward(self, x:Tensor, token_positions:Tensor, return_attention:bool = False):
        """
            From the assignment:
                in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = einx.rearrange("... seq_len (heads d) -> ... heads seq_len d",q,heads=self.num_heads)
        k = einx.rearrange("... seq_len (heads d) -> ... heads seq_len d",k,heads=self.num_heads)
        v = einx.rearrange("... seq_len (heads d) -> ... heads seq_len d",v,heads=self.num_heads)

        # Just like the MultiheadSelfAttention, but with these extra lines:
        mutlihead_token_positions = einx.rearrange("... s -> ... 1 s",token_positions)
        q = self.rope(q,token_positions=mutlihead_token_positions)
        k = self.rope(k,token_positions=mutlihead_token_positions)
        assert isinstance(v,torch.Tensor) # make vscode happier

        triangle_mask = self.make_a_triangle_mask(sequence_length=x.shape[-2],device=x.device)

        attn_output, attn_weights = scaled_dot_product_attention(k=k, q=q, v=v, mask=triangle_mask, return_attention=True)

        attn_output = einx.rearrange("... heads seq d_v -> ... seq (heads d_v)",attn_output)
        output = self.o_proj(attn_output)
        if return_attention:
            return output, attn_weights
        return output
