import json
from pathlib import Path
import sqlite3

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model import (
    NUM_CHAMPIONS_PER_GAME,
    LeagueDraftModel,
)
from model_debug import ModelDebugger

from vocabulary import Vocabulary

BATCH_SIZE = 32
MAX_EPOCHS = 15
LEARNING_RATE = 0.0001

# number of epochs without val_loss improvement to stop training
PATIENCE_EARLY_STOPPING = 3

# number of epochs without meaningful improvement before lowering learning rate
PATIENCE_SCHEDULER = 0

# minimum percentage decrease in val_loss to consider the change meaningful
MINIMUM_THRESHOLD = 0.01

WEIGHT_DECAY = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.5

RANDOM_SEED = 99

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / 'data'
CHECKPOINT_DIRECTORY = PROJECT_ROOT / 'artifacts' / 'checkpoints'
FIGURE_DIRECTORY = PROJECT_ROOT / 'artifacts' / 'figures'

CHAMPION_COLUMNS = [
    'champ_1',
    'champ_2',
    'champ_3',
    'champ_4',
    'champ_5',
    'champ_6',
    'champ_7',
    'champ_8',
    'champ_9',
    'champ_10',
]


class ChampionDataset(Dataset):
    def __init__(self, data, mask_id):
        self.data = data
        self.mask_id = mask_id

    def __len__(self):
        return len(self.data) * NUM_CHAMPIONS_PER_GAME
    
    def __getitem__(self, idx):

        match_id = idx // NUM_CHAMPIONS_PER_GAME
        champ_id = idx % NUM_CHAMPIONS_PER_GAME

        current_match = self.data[match_id]
        masked_champ = current_match[champ_id]
        masked_match = current_match.clone()
        masked_match[champ_id] = self.mask_id
        
        return masked_match, masked_champ

def evaluate(model, loader, loss_fn, device, mask_id):

    model.eval()

    running_loss = 0.0
    num_examples = 0

    with torch.inference_mode():
        for x_v, y_v in loader:

            batch_size = y_v.size(0)
            
            x_v = x_v.to(device, non_blocking=(device.type == 'cuda'))
            y_v = y_v.to(device, non_blocking=(device.type == 'cuda'))

            logits = model(x_v)
            logits = mask_logits(x_v, logits, mask_id)
            loss = loss_fn(logits, y_v)

            running_loss += loss.item() * batch_size
            num_examples += batch_size

    return running_loss / num_examples

def mask_logits(x_b, logits, mask_id):
    '''sets logits from champs already seen in the same game to -inf'''

    # create a mask which sets to false wherever it sees a masked token
    mask = (x_b != mask_id)

    # removes the masked tokens and reshapes the tensor into (BATCH_SIZE, NUM_CHAMPIONS_PER_GAME - 1)
    x_clean = x_b[mask].view(x_b.size(0), x_b.size(1) - 1)

    # sets the logits at the seen champions to -inf for each batch
    output = torch.scatter(logits, dim=1, index = x_clean, value=float('-inf'))
    
    return output

def train_epoch(model, loader, optimizer, loss_fn, device, mask_id):

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
        logits = mask_logits(x_b, logits, mask_id)
        loss = loss_fn(logits, y_b)

        loss.backward()
        
        optimizer.step()

        running_loss += loss.item() * batch_size
        num_examples += batch_size

    return running_loss / num_examples


def save_checkpoint(model, path, champ_names, loss):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'champ_names': champ_names,
    }
    torch.save(checkpoint, path)


def main():
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with open(DATA_DIRECTORY / 'champ_names.json', 'r') as f:
        champ_names = json.load(f)

    conn = sqlite3.connect(DATA_DIRECTORY / 'league_data.db')
    df = pd.read_sql_query("SELECT * FROM matches", conn)
    conn.close()

    vocab = Vocabulary(champ_names)

    # lots of bugs when case is not set to lower due to discrepancies between api and data dragon
    df[CHAMPION_COLUMNS] = df[CHAMPION_COLUMNS].apply(lambda x: x.str.lower())

    champ_data = df.drop(['match_id'], axis=1)

    encoded_matches = torch.tensor(
        champ_data.map(vocab.encode).to_numpy(),
        dtype=torch.long,
    )

    split_generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_matches, val_matches = random_split(
        encoded_matches,
        [0.9, 0.1],
        generator=split_generator,
    )


    mask_id = vocab.mask_id()
    train_data = ChampionDataset(train_matches, mask_id)
    val_data = ChampionDataset(val_matches, mask_id)

    loader_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_load = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type=='cuda'),
        generator=loader_generator,
        drop_last=True,
    )

    val_load = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(device.type=='cuda'),
    )

    model = LeagueDraftModel(total_champions=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=LEARNING_RATE_DECAY_FACTOR, patience=PATIENCE_SCHEDULER, threshold=MINIMUM_THRESHOLD)
    debugger = ModelDebugger(model, optimizer)

    initial_train_loss = evaluate(model, train_load, loss_fn, device, mask_id)
    initial_val_loss = evaluate(model, val_load, loss_fn, device, mask_id)

    debugger.record_epoch(0, initial_train_loss, initial_val_loss)

    best_loss = initial_val_loss
    best_epoch = 0
    patience = 0

    save_checkpoint(model, CHECKPOINT_DIRECTORY / 'best_model.pth', champ_names, best_loss)

    for epoch in range(1, MAX_EPOCHS + 1):

        # train the model and calculate train loss
        train_loss = train_epoch(
            model,
            train_load,
            optimizer,
            loss_fn,
            device,
            mask_id
        )

        # calculate validation loss
        val_loss = evaluate(
            model,
            val_load,
            loss_fn,
            device,
            mask_id
        )

        debugger.record_epoch(epoch, train_loss, val_loss)

        # multiply learning rate by LEARNING_RATE_DECAY_FACTOR if val_loss doesnt improve by a factor of at least MINIMUM_THRESHOLD
        scheduler.step(val_loss)

        # early stopping
        if val_loss > best_loss:
            patience += 1

        else:
            best_loss = val_loss
            best_epoch = epoch
            save_checkpoint(model, CHECKPOINT_DIRECTORY / 'best_model.pth', champ_names, best_loss)
            patience = 0

        if patience == PATIENCE_EARLY_STOPPING:
            print(f'Max patience reached, early stopping. Best model found at epoch {best_epoch}')
            break

        print(f"Epoch {epoch} done")

    debugger.save_figures(best_epoch, FIGURE_DIRECTORY)
    print(f"Saved debugger figures for best epoch {best_epoch} to {FIGURE_DIRECTORY}")


if __name__ == '__main__':
    main()
