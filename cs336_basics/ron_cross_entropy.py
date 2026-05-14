
from .ron_softmax import softmax
import einx
from jaxtyping import Float, Int
import torch
from torch import Tensor

def cross_entropy(
        predicted_logits: Float[Tensor, " batch_size vocab_size"], 
        targets: Int[Tensor, " batch_size"]
    )-> Float[Tensor, ""]:
    """
        Deliverable: 
            Write a function to compute the cross entropy loss, which takes in 
            predicted logits (oi ) and 
            targets (xi+1 ) 
            and computes the cross entropy 
                ℓi = − log softmax(oi )[xi+1 ]. 
            Your function should handle the following:
            • Subtract the largest element for numerical stability.
            • Cancel out log and exp whenever possible.
            • Handle any additional batch dimensions and return the average across the batch. As with sec-
            tion 3.3, we assume batch-like dimensions always come first, before the vocabulary size dimension.
            Implement [adapters.run_cross_entropy], then run uv run pytest -k test_cross_entropy
            to test your implementation.

        Hint -- log softtmax math should match these
            lsm = torch.log_softmax(subtracted,-1)
            lsm = einx.log_softmax("... [c]",subtracted)
        and the whole thing should match
            F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    """

    # Subtract the largest element for numerical stability.
    # Torch 2.6 and old einx
    #maxes,_idxs = einx.max("... [vocab]",predicted_logits) # Why does it choke with ,keepdims=True ?
    # Torch 2.7 and new einx
    maxes = einx.max("... [vocab]",predicted_logits) # Why does it choke with ,keepdims=True ?
    
    maxes = einx.rearrange("... -> ... 1", maxes) # put the dim back
    assert isinstance(maxes,Tensor) # avoid vscode warning
    subtracted = predicted_logits - maxes
    
    # Compute the log softmax
    e = torch.exp(subtracted)
    s = einx.sum("... vocab -> ... 1", e) # einx.sum("... [vocab]", e, keepdims=True)
    lsm = subtracted - torch.log(s)

    # Pick the target values for each target class
    picked = einx.get_at("... [vocab], ... -> ... 1", lsm, targets)
    # Mean over batch-like dimensions
    return -einx.mean("[...]",picked) # torch.mean(picked)
    
"""
More hints: 

    Naively trying: 

        sm = softmax(subtracted,-1)
        lsm = torch.log(sm)


        # # fails on the 
        # # "cross-entropy handles numerical overflow issues" test :(
        # # E       +inf location mismatch:
        # # E        ACTUAL: array(inf)
        # # E        DESIRED: array(272.9625, dtype=float32)

  

    From the unit test: 
                inputs = torch.tensor(
                    [
                        [
                            [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                            [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                            [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                            [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
                        ],
                        [
                            [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                            [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                            [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                            [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
                        ],
                    ]
                )
                targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
                expected = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
        
    Einx hint -- these are quite simlar:
                einx.get_at("... [c], ... -> ...", lsm, targets)
                einx.get_at("... [c], ... -> ... 1", lsm, targets)
                torch.gather(lsm, -1, targets.unsqueeze(-1))
                lsm[torch.arange(targets.shape[0]), targets] # but only 2 dim

            For example, with the input:
            >>> lsm = 100 * torch.arange(2)[:, None, None] + 10 * torch.arange(3)[None,:,None] + torch.arange(5)[None,None,:]; lsm
            tensor([[[  0,   1,   2,   3,   4],
                    [ 10,  11,  12,  13,  14],
                    [ 20,  21,  22,  23,  24]],

                    [[100, 101, 102, 103, 104],
                    [110, 111, 112, 113, 114],
                    [120, 121, 122, 123, 124]]])
            >>> einx.get_at("a b [c], a b -> a b 1",lsm,[[0,1,2],[1,2,3]])
            tensor([[[  0],
                    [ 11],
                    [ 22]],

                    [[101],
                    [112],
                    [123]]])
            >>>  torch.gather(lsm,-1, torch.tensor([[0,1,2],[1,2,3]]).unsqueeze(-1))

"""