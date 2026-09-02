import torch

def mask_logits(x_b, bans, logits):
    '''sets logits from champs already seen in the same game to -inf'''

    # shape (173)
    class_ids = torch.arange(
        logits.size(1),
        device=logits.device
    )

    # shape (1, 1, 173)
    class_ids = class_ids.view(1, 1, -1)

    # shape: (B, 10, 1) -> (B, 10, 1) == (1, 1, 173) -> (B, 10, 173) -> any(1) -> (B, 173)
    # for each picked champion, there will be 173 elements with all false and a single True for the champion that was picked there. any(1) then combines all these trues in the 
    # the 2nd dimension which is the 'match' dimension so there will be up to 9 Trues (masked token is always false) out of 173 corresponding to all champions picked that game. 
    seen = (x_b.unsqueeze(-1) == class_ids).any(1)

    # similarly for banned
    banned = (bans.unsqueeze(-1) == class_ids).any(1)

    # shape(B, 173)
    blocked = seen | banned

    output = logits.masked_fill(blocked, float('-inf'))
    
    return output