import torch
import numpy as np
import os
import pickle
import pandas as pd
import ast
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm  # 진행률 표시
from model import NeuMF 

# ==========================================
# 1. 평가 설정 
# ==========================================
DATA_DIR = 'data/'
EMBED_DIM = 64
TOP_K = 100 # 카카오 아레나 리더보드 기준
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  메모리 조절 파라미터
USER_BATCH_SIZE = 1     # 한 번에 n명의 유저 평가
ITEM_BATCH_SIZE = 2000  # 한 번에 계산할 후보 곡 수 

# ==========================================
# 2. 평가용 데이터셋 클래스
# ==========================================
class FullEvalDataset(Dataset):
    def __init__(self, test_path):
        print(f" 평가 데이터 로드 ({test_path})")
        self.data = pd.read_json(test_path, orient='records')
        
        # 문자열로 저장된 리스트 파싱 
        cols_to_parse = ['songs', 'answers', 'tags', 'artists', 'genres']
        for col in cols_to_parse:
            if col in self.data.columns and len(self.data) > 0 and isinstance(self.data[col].iloc[0], str):
                self.data[col] = self.data[col].apply(ast.literal_eval)
        
        print(f"총 {len(self.data)}명의 유저 평가 준비")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 모델 입력에 필요한 모든 Feature 추출
        history = row['songs']
        tags = row['tags'] if 'tags' in row and isinstance(row['tags'], list) else []
        artists = row['artists'] if 'artists' in row and isinstance(row['artists'], list) else []
        genres = row['genres'] if 'genres' in row and isinstance(row['genres'], list) else []
        answers = row['answers'] if isinstance(row['answers'], list) else []

        # 빈 리스트 패딩 처리
        if len(tags) == 0: tags = [0]
        if len(artists) == 0: artists = [0]
        if len(genres) == 0: genres = [0]

        return {
            'history': torch.tensor(history, dtype=torch.long),
            'tags': torch.tensor(tags, dtype=torch.long),
            'artists': torch.tensor(artists, dtype=torch.long),
            'genres': torch.tensor(genres, dtype=torch.long),
            'answers': answers  # 정답은 텐서로 변환하지 않음 
        }

def collate_fn_full(batch):
    # User History Padding
    history_padded = pad_sequence([i['history'] for i in batch], batch_first=True, padding_value=0)
    tags_padded = pad_sequence([i['tags'] for i in batch], batch_first=True, padding_value=0)
    artists_padded = pad_sequence([i['artists'] for i in batch], batch_first=True, padding_value=0)
    genres_padded = pad_sequence([i['genres'] for i in batch], batch_first=True, padding_value=0)
    
    answers = [i['answers'] for i in batch]
    return history_padded, tags_padded, artists_padded, genres_padded, answers

# ==========================================
# 3. 평가 메인 로직
# ==========================================
def evaluate():
    print(f"Evaluation Device: {DEVICE}")

    # 메타 데이터 로드 (ID 개수 파악용)
    matrix_path = os.path.join(DATA_DIR, 'song_embed_matrix.npy')
    map_path = os.path.join(DATA_DIR, 'id_maps.pkl')
    
    if not os.path.exists(matrix_path) or not os.path.exists(map_path):
        print("Error: 메타 데이터 파일이 없습니다.")
        return

    with open(map_path, 'rb') as f:
        maps = pickle.load(f)
    
    song_matrix = np.load(matrix_path)
    num_songs = song_matrix.shape[0]
    num_tags = len(maps.get('tag', {})) + 1
    num_artists = len(maps.get('artist', {})) + 1
    num_genres = len(maps.get('genre', {})) + 1
    
    # 평가 대상: 전체 아이템 ID (0 ~ num_songs-1)
    all_item_ids = torch.arange(num_songs, dtype=torch.long).to(DEVICE)

    # 모델 로드 및 초기화
    # 체크포인트 파일명으로 수정
    checkpoint_path = 'checkpoint_epoch_50.pth' 
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: 모델 파일({checkpoint_path})을 찾을 수 없습니다.")
        return

    print(" NeuMF 모델 로드")
    model = NeuMF(
        num_items=num_songs, num_tags=num_tags, num_artists=num_artists, 
        num_genres=num_genres, embed_dim=EMBED_DIM, pretrained_weights=None
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    #  데이터 로더
    eval_ds = FullEvalDataset(test_path=os.path.join(DATA_DIR, 'test_data.json'))
    eval_loader = DataLoader(eval_ds, batch_size=USER_BATCH_SIZE, shuffle=False, collate_fn=collate_fn_full)

    # 전수 조사 루프
    total_recall = 0.0
    total_ndcg = 0.0
    valid_count = 0
    
    print(f"전수 조사 시작 (Top-{TOP_K})")
    print(f"총 {num_songs}개 곡에 대해 연산 수행 (User당 약 {num_songs/ITEM_BATCH_SIZE:.0f}회 반복)")

    with torch.no_grad():
        # tqdm으로 진행 상황 표시
        for history, tags, artists, genres, answers in tqdm(eval_loader, desc="Evaluating Users"):
            history = history.to(DEVICE)   # [1, Seq]
            tags = tags.to(DEVICE)         # [1, Seq]
            artists = artists.to(DEVICE)   # [1, Seq]
            genres = genres.to(DEVICE)     # [1, Seq]
            
            true_answers = answers[0]      # 현재 유저의 정답 (List)
            if not true_answers: continue  # 정답 없으면 건너뜀

            # 2000개씩 잘라서 계산   
            user_scores = [] # 70만 개 점수를 담을 리스트
            
            for i in range(0, num_songs, ITEM_BATCH_SIZE):
                # 아이템 청크 준비 
                item_chunk = all_item_ids[i : i + ITEM_BATCH_SIZE]
                chunk_size = item_chunk.shape[0]

                # 유저 1명의 정보를 아이템 청크 개수(2000개)만큼 복사
                hist_exp = history.repeat_interleave(chunk_size, dim=0)
                tags_exp = tags.repeat_interleave(chunk_size, dim=0)
                art_exp = artists.repeat_interleave(chunk_size, dim=0)
                gen_exp = genres.repeat_interleave(chunk_size, dim=0)
                
                # NeuMF 모델
                chunk_score = model(hist_exp, tags_exp, art_exp, gen_exp, item_chunk)
                user_scores.append(chunk_score)
            
            # 전체 점수 합치기
            final_scores = torch.cat(user_scores) # [num_songs]
            
            # 이미 들은 곡 제외 
            watched_items = history[0].unique()
            watched_items = watched_items[watched_items != 0] # 패딩 제외
            final_scores[watched_items] = -float('inf')

            # Top-K 추출
            _, topk_indices = torch.topk(final_scores, k=TOP_K)
            pred_list = topk_indices.cpu().numpy().tolist()
            
            # Recall & NDCG
            hit_cnt = 0
            dcg = 0.0
            idcg = 0.0
            
            true_set = set(true_answers)
            
            # DCG 계산
            for rank, item_id in enumerate(pred_list):
                if item_id in true_set:
                    hit_cnt += 1
                    dcg += 1.0 / np.log2(rank + 2) # rank는 0부터 시작하므로 +2
            
            # IDCG 계산
            for rank in range(min(len(true_set), TOP_K)):
                idcg += 1.0 / np.log2(rank + 2)
                
            total_recall += hit_cnt / len(true_set)
            if idcg > 0:
                total_ndcg += dcg / idcg
                
            valid_count += 1

    # 최종 결과 출력
    if valid_count > 0:
        print(f"\nFinal Result (Full Ranking - NeuMF MLP)")
        print(f"Recall@{TOP_K} : {total_recall / valid_count:.4f}")
        print(f"nDCG@{TOP_K}   : {total_ndcg / valid_count:.4f}")
    else:
        print("평가할 유효한 데이터가 없습니다.")

if __name__ == '__main__':
    evaluate()