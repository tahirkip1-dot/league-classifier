import torch

from vocabulary import Vocabulary
from train import mask_logits

def inference(model: torch.nn.Module, names: list[str], bans: list[str], vocab: Vocabulary, device, k = 1):
    '''names must be in the form of a list with champion names and a masked token, in the correct role order of TOP JG MID ADC SUP.
    returns the top k inferences'''

    model.eval()

    with torch.inference_mode():
        encode_names = [vocab.name_to_id(name) for name in names]
        encode_bans = [vocab.name_to_id(ban) for ban in bans]

        # need to add batch dimension
        input = torch.tensor(encode_names, dtype=torch.long).unsqueeze(0)
        input_bans = torch.tensor(encode_bans, dtype=torch.long).unsqueeze(0)

        input = input.to(device, non_blocking=(device.type == 'cuda'))
        input_bans = input_bans.to(device, non_blocking=(device.type == 'cuda'))

        logits = model(input)
        logits = mask_logits(input, input_bans, logits)

        # returns value indices pairs, only care about index
        _, preds_enc = torch.topk(logits, k=k ,dim=1)

        # remove the batch dimension first with squeeze
        preds = [vocab.id_to_name(pred) for pred in preds_enc.squeeze(0).tolist()]

    return preds