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

MODEL_SEED = 99
LOADER_SEED = 99
SPLIT_SEED = 99

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / 'data'
CHECKPOINT_DIRECTORY = PROJECT_ROOT / 'artifacts' / 'checkpoints'
FIGURE_DIRECTORY = PROJECT_ROOT / 'artifacts' / 'figures'


class ChampionDataset(Dataset):
    def __init__(self, data, mask_id):
        self.data = data
        self.mask_id = mask_id

    def __len__(self):
        return len(self.data) * NUM_CHAMPIONS_PER_GAME
    
    def __getitem__(self, idx):

        match_id = idx // NUM_CHAMPIONS_PER_GAME
        champ_id = idx % NUM_CHAMPIONS_PER_GAME

        current_picks_bans = self.data[match_id]
        current_picks = current_picks_bans[:NUM_CHAMPIONS_PER_GAME]
        current_bans = current_picks_bans[NUM_CHAMPIONS_PER_GAME:]
        masked_champ = current_picks[champ_id]
        masked_match = current_picks.clone()
        masked_match[champ_id] = self.mask_id
        
        return masked_match, current_bans, masked_champ

def evaluate(model, loader, loss_fn, device):

    model.eval()

    running_loss = 0.0
    num_examples = 0

    with torch.inference_mode():
        for x_v, ban_v, y_v in loader:

            batch_size = y_v.size(0)
            
            x_v = x_v.to(device, non_blocking=(device.type == 'cuda'))
            ban_v = ban_v.to(device, non_blocking=(device.type == 'cuda'))
            y_v = y_v.to(device, non_blocking=(device.type == 'cuda'))

            logits = model(x_v)
            logits = mask_logits(x_v, ban_v, logits)
            loss = loss_fn(logits, y_v)

            running_loss += loss.item() * batch_size
            num_examples += batch_size

    return running_loss / num_examples

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

def train_epoch(model, loader, optimizer, loss_fn, device):

    model.train()

    running_loss = 0.0
    num_examples = 0
    for x_b, ban_b, y_b in loader:

        batch_size = y_b.size(0)

        # move batches into gpu
        x_b = x_b.to(device, non_blocking=(device.type == 'cuda'))
        ban_b = ban_b.to(device, non_blocking=(device.type == 'cuda'))
        y_b = y_b.to(device, non_blocking=(device.type == 'cuda'))

        optimizer.zero_grad(set_to_none=True)

        logits = model(x_b)
        logits = mask_logits(x_b, ban_b, logits)
        loss = loss_fn(logits, y_b)

        loss.backward()
        
        optimizer.step()

        running_loss += loss.item() * batch_size
        num_examples += batch_size

    return running_loss / num_examples


def save_checkpoint(model, path, riot_id_to_name, loss):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'riotid_to_name': riot_id_to_name,
    }
    torch.save(checkpoint, path)


def main():
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with open(DATA_DIRECTORY / 'championid_to_name.json') as f:
        champid_to_names = json.load(f)

    conn = sqlite3.connect(DATA_DIRECTORY / 'league_data.db')
    picks_id = pd.read_sql_query("SELECT * FROM matches", conn)
    bans_id = pd.read_sql_query("SELECT * FROM bans", conn)
    conn.close()

    # convert str to ints
    champid_to_names = {int(item[0]):item[1] for item in champid_to_names.items()}

    vocab = Vocabulary(champid_to_names)

    # join each teams picks and bans
    picks_bans_id = picks_id.merge(bans_id, how='inner', on=['match_id', 'team_id'])

    # join teams from the same game
    complete_id = picks_bans_id[picks_bans_id['team_id'] == 100].merge(picks_bans_id[picks_bans_id['team_id'] == 200], how='inner', on='match_id')

    champ_data_id = complete_id.drop(['match_id', 'team_id_x', 'patch_x', 'team_id_y', 'patch_y'], axis=1)

    # reorder into 10 picks then 10 bans
    champ_data_id = champ_data_id[['top_x', 'jungle_x', 'mid_x', 'bot_x', 'support_x', 'top_y', 'jungle_y', 'mid_y', 'bot_y', 'support_y', 'ban_1_x', 'ban_2_x', 'ban_3_x', 'ban_4_x', 'ban_5_x', 'ban_1_y', 'ban_2_y', 'ban_3_y', 'ban_4_y', 'ban_5_y']]


    encoded_matches = torch.tensor(
        champ_data_id.map(vocab.riot_id_to_id).to_numpy(),
        dtype=torch.long,
    )

    split_generator = torch.Generator().manual_seed(SPLIT_SEED)

    train_matches, val_matches = random_split(
        encoded_matches,
        [0.9, 0.1],
        generator=split_generator,
    )


    mask_id = vocab.mask_id
    train_data = ChampionDataset(train_matches, mask_id)
    val_data = ChampionDataset(val_matches, mask_id)

    loader_generator = torch.Generator().manual_seed(LOADER_SEED)
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

    model = LeagueDraftModel(total_champions=len(vocab), mask_id=mask_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=LEARNING_RATE_DECAY_FACTOR, patience=PATIENCE_SCHEDULER, threshold=MINIMUM_THRESHOLD)
    debugger = ModelDebugger(model, optimizer)

    initial_train_loss = evaluate(model, train_load, loss_fn, device)
    initial_val_loss = evaluate(model, val_load, loss_fn, device)

    debugger.record_epoch(0, initial_train_loss, initial_val_loss)

    best_loss = initial_val_loss
    best_epoch = 0
    patience = 0

    save_checkpoint(model, CHECKPOINT_DIRECTORY / 'best_model.pth', champid_to_names, best_loss)

    for epoch in range(1, MAX_EPOCHS + 1):

        # train the model and calculate train loss
        train_loss = train_epoch(
            model,
            train_load,
            optimizer,
            loss_fn,
            device,
        )

        # calculate validation loss
        val_loss = evaluate(
            model,
            val_load,
            loss_fn,
            device,
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
            save_checkpoint(model, CHECKPOINT_DIRECTORY / 'best_model.pth', champid_to_names, best_loss)
            patience = 0

        if patience == PATIENCE_EARLY_STOPPING:
            print(f'Max patience reached, early stopping. Best model found at epoch {best_epoch}')
            break

        print(f"Epoch {epoch} done")

    debugger.save_figures(best_epoch, FIGURE_DIRECTORY)
    print(f"Saved debugger figures for best epoch {best_epoch} to {FIGURE_DIRECTORY}")


if __name__ == '__main__':
    main()
