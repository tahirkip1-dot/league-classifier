import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import sqlite3

from evaluate import evaluate
from model import (
    NUM_CHAMPIONS_PER_GAME,
    LeagueDraftModel,
    encode,
)
from model_debug import ModelDebugger

BATCH_SIZE = 32
MAX_EPOCHS = 15
LEARNING_RATE = 0.0001
MAX_PATIENCE = 3


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


conn = sqlite3.connect('../data/league_data.db')
df = pd.read_sql_query("SELECT * FROM matches", conn)

# lots of bugs when case is not set to lower due to discrepancies between api and data dragon
df[['champ_1', 'champ_2', 'champ_3', 'champ_4', 'champ_5', 'champ_6', 'champ_7', 'champ_8', 'champ_9', 'champ_10']] = df[['champ_1', 'champ_2', 'champ_3', 'champ_4', 'champ_5', 'champ_6', 'champ_7', 'champ_8', 'champ_9', 'champ_10']].apply(lambda x: x.str.lower())

conn.close()


champ_data = df.drop(['match_id'], axis=1)


encoded_matches = torch.tensor(
    champ_data.map(encode).to_numpy(),
    dtype=torch.long,
)


class ChampionDataset(Dataset):
    def __init__(self, data, masked_encoded):
        self.data = data
        self.masked_id = masked_encoded

    ''' two options here, not sure which is better. 
    1. mask every champion from every game 
    2. mask x amount of randomly chosen champion every game
    currently implementing mask every champion to maximise data
    '''

    '''  exactly one champion is chosen uniformly at random each game and masked
    def __len__(self):
            return len(self.data)

    def __getitem__(self, idx):
        current_match = self.data.iloc[idx]
        masked_match = current_match.copy()

        masked_id = torch.randint(low=0, high=NUM_CHAMPIONS_PER_GAME, size=(1,)).item()
        masked_champ = masked_match.iat[masked_id]
        masked_match.iloc[masked_id] = 'masked'

        x_df = masked_match.apply(self.encode)
        x_np = x_df.to_numpy(dtype = np.int64)

        return torch.tensor(x_np), torch.tensor(self.encode(masked_champ))
    '''
    # every champion is masking exactly once

    def __len__(self):
        return len(self.data) * NUM_CHAMPIONS_PER_GAME
    
    def __getitem__(self, idx):

        match_id = idx // 10
        champ_id = idx % 10

        current_match = self.data[match_id]
        masked_champ = current_match[champ_id]
        masked_match = current_match.clone()
        masked_match[champ_id] = self.masked_id
        
        return masked_match, masked_champ


generator = torch.Generator().manual_seed(42)

train_matches, val_matches = random_split(encoded_matches, [0.9, 0.1], generator=generator)


train_data = ChampionDataset(train_matches, encode('masked'))
val_data = ChampionDataset(val_matches, encode('masked'))


train_load = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type=='cuda'))
val_load = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type=='cuda'))


model = LeagueDraftModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

debugger = ModelDebugger(model)


def train_epoch(model, loader, optimizer, loss_fn, device):

    model.train()

    running_loss = 0.0
    num_examples = 0
    for x_b, y_b in loader:

        batch_size = y_b.size(0)

        # move batches into gpu
        x_b = x_b.to(device, non_blocking=(device.type == 'cuda'))
        y_b = y_b.to(device, non_blocking=(device.type == 'cuda'))

        optimizer.zero_grad(set_to_none=True)

        logits = model(x_b)
        loss = loss_fn(logits, y_b)

        loss.backward()
        
        optimizer.step()

        running_loss += loss.item() * batch_size
        num_examples += batch_size

    return running_loss / num_examples


initial_train_loss = evaluate(model, train_load, loss_fn, device)
initial_val_loss = evaluate(model, val_load, loss_fn, device)

debugger.record_epoch(0, initial_train_loss, initial_val_loss)

best_loss = initial_val_loss


patience = 0

for epoch in range(1, MAX_EPOCHS + 1):

    # train the model and calculate train loss
    train_loss = train_epoch(
        model,
        train_load,
        optimizer,
        loss_fn,
        device
    )

    # calculate validation loss
    val_loss = evaluate(
        model,
        val_load,
        loss_fn,
        device
    )

    debugger.record_epoch(epoch, train_loss, val_loss)

    # early stopping
    if val_loss > best_loss:
        patience += 1

    else:
        best_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')
        patience = 0

    if patience == MAX_PATIENCE:
        print(f'Max patience reached, early stopping. Best model found at epoch {best_epoch}')
        break
    
    print(f"Epoch {epoch} done")


debugger.plot_losses()


debugger.plot_distributions(best_epoch)
