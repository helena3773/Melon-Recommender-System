import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import ast

class MelonDataset(Dataset):
    def __init__(self, file_path, num_songs, mode='train', num_negatives=4):
        self.mode = mode
        self.num_songs = num_songs
        self.num_negatives = num_negatives
        
        print(f"[{mode.upper()}] 데이터 로딩 ({file_path})")
        self.data = pd.read_json(file_path, orient='records')
        
        cols_to_parse = ['songs', 'tags', 'artists', 'genres']
        if mode != 'train':
            cols_to_parse.append('answers')
            
        for col in cols_to_parse:
            if col in self.data.columns and len(self.data) > 0 and isinstance(self.data[col].iloc[0], str):
                self.data[col] = self.data[col].apply(ast.literal_eval)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        

        history = row['songs']
        tags = row['tags'] if 'tags' in row and isinstance(row['tags'], list) else []
        artists = row['artists'] if 'artists' in row and isinstance(row['artists'], list) else []
        genres = row['genres'] if 'genres' in row and isinstance(row['genres'], list) else []
        
        feats = [
            row.get('scaled_like', 0.0),      # 좋아요
            row.get('scaled_song_cnt', 0.0),  # 곡 수
            row.get('avg_year', 0.5)          # 발매연도
        ]
        feats_tensor = torch.tensor(feats, dtype=torch.float)

        if len(tags) == 0: tags = [0]
        if len(artists) == 0: artists = [0]
        if len(genres) == 0: genres = [0]
        
        if self.mode == 'train':
            if len(history) < 2:
                pos_item = history[0]
                input_seq = history
            else:
                target_idx = np.random.randint(0, len(history))
                pos_item = history[target_idx]
                input_seq = history[:target_idx] + history[target_idx+1:]
            
            neg_items = []
            while len(neg_items) < self.num_negatives:
                neg = np.random.randint(0, self.num_songs)
                if neg not in history:
                    neg_items.append(neg)

            return {
                'history': torch.tensor(input_seq, dtype=torch.long),
                'pos_item': torch.tensor(pos_item, dtype=torch.long),
                'neg_items': torch.tensor(neg_items, dtype=torch.long),
                'tags': torch.tensor(tags, dtype=torch.long),
                'artists': torch.tensor(artists, dtype=torch.long),
                'genres': torch.tensor(genres, dtype=torch.long),
                'feats': feats_tensor
            }
        else:
            answers = row['answers'] if isinstance(row['answers'], list) else []
            return {
                'history': torch.tensor(history, dtype=torch.long),
                'answers': answers,
                'tags': torch.tensor(tags, dtype=torch.long),
                'artists': torch.tensor(artists, dtype=torch.long),
                'genres': torch.tensor(genres, dtype=torch.long),
                'feats': feats_tensor
            }

def collate_fn(batch):
    history_padded = pad_sequence([i['history'] for i in batch], batch_first=True, padding_value=0)
    tags_padded = pad_sequence([i['tags'] for i in batch], batch_first=True, padding_value=0)
    artists_padded = pad_sequence([i['artists'] for i in batch], batch_first=True, padding_value=0)
    genres_padded = pad_sequence([i['genres'] for i in batch], batch_first=True, padding_value=0)

    feats_stacked = torch.stack([i['feats'] for i in batch])
    
    batch_data = {
        'history': history_padded,
        'tags': tags_padded,
        'artists': artists_padded,
        'genres': genres_padded,
        'feats': feats_stacked
    }
    
    if 'pos_item' in batch[0]: # Train
        batch_data['pos_item'] = torch.stack([i['pos_item'] for i in batch])
        batch_data['neg_items'] = torch.stack([i['neg_items'] for i in batch])
    else: # Val
        batch_data['answers'] = [i['answers'] for i in batch]
        
    return batch_data