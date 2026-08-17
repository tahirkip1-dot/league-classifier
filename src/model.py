import torch
import torch.nn as nn


NUM_CHAMPIONS_PER_GAME = 10
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

NUM_ATTENTION_HEADS = 4

DROPOUT_RATE = 0.1


class LeagueDraftModel(nn.Module):
    def __init__(self, total_champions, mask_id):
        
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=total_champions, embedding_dim=EMBEDDING_DIM)
        self.role_embedding = nn.Embedding(num_embeddings=NUM_CHAMPIONS_PER_GAME, embedding_dim=EMBEDDING_DIM)
        self.attention = nn.MultiheadAttention(EMBEDDING_DIM, num_heads=NUM_ATTENTION_HEADS, batch_first=True)
        self.layernorm = nn.LayerNorm(EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.lm = nn.Linear(EMBEDDING_DIM * NUM_CHAMPIONS_PER_GAME, HIDDEN_DIM)

        # dont include masked token as a possible output
        self.output_layer = nn.Linear(HIDDEN_DIM, total_champions - 1)

        # store role ids here instead of creating a new tensor every loop. ally team is always 01234 in order of TOP JG MID ADC SUP.
        self.register_buffer(
            'role_ids',
            torch.tensor([
                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
            ]),
            persistent = False
        )
        
        self.mask_id = mask_id


    def forward(self, x):

        # x is of shape (BATCH_SIZE, NUM_CHAMPIONS_PER_GAME)
        batch_size = x.size(0)
        token_embeds = self.token_embedding(x) # Shape: (BATCH_SIZE, NUM_CHAMPIONS_PER_GAME, EMBEDDING_DIM)
        
        # finds where the mask_token is in each batch. unsqueeze is required to broadcast in the next step
        where_mask = (x==self.mask_id).nonzero(as_tuple=True)[1].unsqueeze(1) # Shape: (BATCH_SIZE, 1)
        
        # determines where the mask is and duplicates the desired role_ids. torch.where broadcasts to the 2nd dimension
        role_ids = torch.where((where_mask > NUM_CHAMPIONS_PER_GAME // 2 - 1), self.role_ids[1], self.role_ids[0]) # Shape: (BATCH_SIZE, NUM_CHAMPIONS_PER_GAME)
        
        role_embeds = self.role_embedding(role_ids) # Shape: (BATCH_SIZE, NUM_CHAMPIONS_PER_GAME, EMBEDDING_DIM)
        embeds = token_embeds + role_embeds # Shape: (BATCH_SIZE, NUM_CHAMPIONS_PER_GAME, EMBEDDING_DIM)

        # attention
        attended, _ = self.attention(query=embeds, key=embeds, value=embeds, need_weights=False) # Shape: (BATCH_SIZE, NUM_CHAMPIONS_PER_GAME, EMBEDDING_DIM)

        # dropout
        attended = self.dropout(attended)

        # residual
        attended = attended + embeds

        # layer norm
        attended = self.layernorm(attended) # Shape: (BATCH_SIZE, NUM_CHAMPIONS_PER_GAME, EMBEDDING_DIM)

        # flatten
        attended = attended.reshape(batch_size, NUM_CHAMPIONS_PER_GAME * EMBEDDING_DIM) # Shape: (BATCH_SIZE, NUM_CHAMPIONS_PER_GAME * EMBEDDING_DIM)

        pre_activations = torch.tanh(self.lm(attended))  # Shape: (BATCH_SIZE, HIDDEN_DIM)

        # dropout
        pre_activations = self.dropout(pre_activations)

        logits = self.output_layer(pre_activations) # Shape: (BATCH_SIZE, TOTAL_CHAMPIONS - 1)
        return logits
