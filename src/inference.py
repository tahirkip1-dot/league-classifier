import torch

from model import LeagueDraftModel, decode, encode


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def inference(model, names, device, k = 1):
    '''names must be in the form of a list with champion names and a masked token, in the correct role order of TOP JG MID ADC SUP
    returns the top k inferences'''

    model.eval()

    with torch.inference_mode():
        encode_names = [encode(name) for name in names]

        # need to add batch dimension
        input = torch.tensor(encode_names, dtype=torch.long).unsqueeze(0)

        input = input.to(device, non_blocking=(device.type == 'cuda'))
        logits = model(input)

        # returns value indices pairs, only care about index
        _, preds_enc = torch.topk(logits, k=k ,dim=1)

        # remove the batch dimension first with squeeze
        preds = [decode(pred) for pred in preds_enc.squeeze(0).tolist()]

    return preds


# do some inference
model = LeagueDraftModel().to(device)
model.load_state_dict(torch.load('best_model.pth'))
match1 = ['aatrox', 'sejuani', 'orianna', 'masked', 'lulu', 'ornn', 'monkeyking', 'viktor', 'ashe', 'nautilus']
guess = inference(model, match1, device, 7)
print(guess)
