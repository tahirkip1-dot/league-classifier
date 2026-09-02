import sqlite3
import pandas as pd
import torch
from torch.utils.data import Dataset
from model import NUM_CHAMPIONS_PER_GAME

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

def load_matches(db_path, vocab):
    '''returns all matches found in db_path encoded as a tensor ready for dataset object creation'''
    
    conn = sqlite3.connect(db_path)
    picks_id = pd.read_sql_query("SELECT * FROM matches", conn)
    bans_id = pd.read_sql_query("SELECT * FROM bans", conn)
    conn.close()

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

    return encoded_matches