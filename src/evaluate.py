import torch

def evaluate(model, loader, loss_fn, device):

    model.eval()

    running_loss = 0.0
    num_examples = 0

    with torch.inference_mode():
        for x_v, y_v in loader:

            batch_size = y_v.size(0)
            
            x_v = x_v.to(device, non_blocking=(device.type == 'cuda'))
            y_v = y_v.to(device, non_blocking=(device.type == 'cuda'))

            logits = model(x_v)
            loss = loss_fn(logits, y_v)

            running_loss += loss.item() * batch_size
            num_examples += batch_size

    return running_loss / num_examples
