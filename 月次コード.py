import pandas as pd
import numpy as np
import os
import glob
import joblib
import lightgbm as lgb
from datetime import datetime
import time
from tqdm import tqdm
import warnings

# =====================================================================
# 🌟 修正必須1＆2対応済：Colab環境の初期化とドライブマウント
# =====================================================================
os.environ['TZ'] = 'Asia/Tokyo'
time.tzset()
warnings.filterwarnings('ignore')

print("Googleドライブをマウントします...")
from google.colab import drive
drive.mount('/content/drive')

WORK_DIR = './KeirinData'
os.makedirs(WORK_DIR, exist_ok=True)

# =====================================================================
# ユーティリティ関数
# =====================================================================
def get_latest_file(pattern):
    """指定されたパターンのファイル群から、名前順で最新のものを取得する"""
    files = glob.glob(os.path.join(WORK_DIR, pattern))
    if not files: return None
    return sorted(files)[-1]

# =====================================================================
# 🌟 断絶解消ロジック：日次の生データ(Wide)をAI学習用(Long)に変換し、正解ラベルを作る
# =====================================================================
def generate_training_features(df_raw):
    records = []
    # 欠損値等をあらかじめ埋めておく
    df_raw['line_prediction'] = df_raw['line_prediction'].fillna('-')
    
    for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="🔄 AI学習用データに変換中 (Wide -> Long)"):
        race_id = f"{row.get('date', '')}_{row.get('place_name', '')}_{row.get('race_num', '')}"
        line_str = str(row['line_prediction'])
        lines = []
        if line_str and line_str != "-" and line_str != "nan":
            line_str = line_str.replace("←", "").strip()
            parts = [p.strip() for p in line_str.split("|")]
            for p in parts:
                cars = [c for c in p.split() if c.isdigit()]
                if cars: lines.append(cars)
        total_lines = len(lines)
        
        players = {}
        for i in range(1, 8):
            prefix = f"c{i}"
            if row.get(f"{prefix}_existence", 0) == 1:
                try: score = float(row.get(f"{prefix}_score", 0))
                except: score = 0.0
                try: b_count = float(row.get(f"{prefix}_b", 0))
                except: b_count = 0.0
                
                # 🎯 学習用の正解ラベル（target）を生成：1着=0, 2着=1, 3着=2, 4着以下・失格=3
                rank_val = str(row.get(f"{prefix}_rank", "0"))
                if rank_val == "1": target = 0
                elif rank_val == "2": target = 1
                elif rank_val == "3": target = 2
                else: target = 3

                players[str(i)] = {
                    'score': score,
                    'b_count': b_count,
                    'leg': str(row.get(f"{prefix}_leg", "")),
                    'area': str(row.get(f"{prefix}_area", "")),
                    'target': target
                }
                
        if not players: continue
        
        max_score = max([p['score'] for p in players.values()])
        chaos_idx = np.std([p['score'] for p in players.values()])
        
        # 配当金の取得 (オッズAIの学習用)
        try: payout_yen = float(row.get('payout_yen', 0))
        except: payout_yen = 0.0
        
        for car_num, p_info in players.items():
            position_in_line, line_length, is_solo, leader_car = 0, 1, 1, car_num
            for l in lines:
                if car_num in l:
                    position_in_line = l.index(car_num) + 1
                    line_length = len(l)
                    is_solo = 1 if line_length == 1 else 0
                    leader_car = l[0]
                    break
                    
            leader_score = players.get(leader_car, {}).get('score', 0)
            leader_b_count = players.get(leader_car, {}).get('b_count', 0)
            is_same_area = 1 if p_info['area'] == players.get(leader_car, {}).get('area', "") and car_num != leader_car else 0
            
            records.append({
                'race_id': race_id, 'date': row.get('date', ''), 'place_name': row.get('place_name', ''), 'race_num': row.get('race_num', ''),
                'car_number': int(car_num), 'score': p_info['score'], 'b_count': p_info['b_count'],
                'score_diff_from_max': max_score - p_info['score'], 'position_in_line': position_in_line,
                'line_length': line_length, 'is_solo': is_solo, 'leader_score': leader_score,
                'leader_b_count': leader_b_count, 'is_same_area_as_leader': is_same_area,
                'chaos_idx': chaos_idx, 'bank_length': str(row.get('bank_length', '400')), 'leg': p_info['leg'],
                'total_lines': total_lines, 
                'target': p_info['target'], 'payout_yen': payout_yen # 🎯 AI学習用の重要カラム
            })
            
    return pd.DataFrame(records)

# =====================================================================
# 月次学習パイプライン メイン処理
# =====================================================================
def run_monthly_training_pipeline():
    current_ym = datetime.now().strftime('%Y%m')
    print(f"\n=== 🔄 月次AIアップデート・パイプライン開始 (対象月: {current_ym}) ===")
    
    # 1. マスターファイルの自動検知
    master_file = get_latest_file('kdreams_analysis_*_master.csv')
    if not master_file:
        print("❌ エラー: 日次バッチで作成されたマスターデータが見つかりません。")
        return

    print(f">> 📂 マスターデータを読み込み中: {os.path.basename(master_file)}")
    df_raw = pd.read_csv(master_file, low_memory=False)
    
    # 2. 特徴量と正解ラベルの生成 (Wide -> Long)
    df = generate_training_features(df_raw)
    
    if df.empty:
        print("❌ エラー: 学習可能なデータが抽出できませんでした。")
        return

    # =====================================================================
    # 3. 閾値（足切り基準）の計算と保存
    # =====================================================================
    print("\n>> 📊 最新の「黄金の乱戦」判定ルール（閾値）を計算中...")
    race_stats = df.groupby('race_id')['score'].std().reset_index(name='score_std').dropna()
    b_stats = df.groupby('race_id')['b_count'].sum().reset_index(name='b_sum').dropna()
    
    score_bins = [-np.inf, race_stats['score_std'].quantile(0.33), race_stats['score_std'].quantile(0.66), np.inf]
    b_bins = [-np.inf, b_stats['b_sum'].quantile(0.33), b_stats['b_sum'].quantile(0.66), np.inf]
    
    thresholds = {
        'score_bins': score_bins,
        'b_bins': b_bins,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    threshold_file = os.path.join(WORK_DIR, f'keirin_thresholds_{current_ym}.pkl')
    joblib.dump(thresholds, threshold_file)
    print(f"   ✅ 閾値ファイルを保存しました: {os.path.basename(threshold_file)}")

    # =====================================================================
    # 4. 勝率予測AIの再学習と保存
    # =====================================================================
    print("\n>> 🚀 勝率予測AIモデルを最新データで再学習中...")
    df['score_gap_type'] = pd.cut(df['race_id'].map(race_stats.set_index('race_id')['score_std']), bins=thresholds['score_bins'], labels=['拮抗(小)', '普通(中)', '鉄板(大)'])
    df['b_sum_type'] = pd.cut(df['race_id'].map(b_stats.set_index('race_id')['b_sum']), bins=thresholds['b_bins'], labels=['先行不在(少)', '標準(中)', 'モガキ合い(多)'])
    
    # 大穴条件のレースのみを抽出
    golden_df = df[(df['score_gap_type'] == '拮抗(小)') & (df['b_sum_type'] == 'モガキ合い(多)') & (df['total_lines'] >= 4)].copy()

    if len(golden_df) < 100:
        print("⚠️ 警告: 学習データ（黄金の乱戦）が少なすぎます。学習を中止します。")
        return

    features_win = ['score', 'b_count', 'score_diff_from_max', 'position_in_line', 'line_length', 'is_solo', 'leader_score', 'leader_b_count', 'is_same_area_as_leader', 'chaos_idx', 'bank_length', 'leg']
    for col in ['bank_length', 'leg']: 
        golden_df[col] = golden_df[col].astype('category')
    
    X_win = golden_df[features_win]
    y_win = golden_df['target'] # 0(1着), 1(2着), 2(3着), 3(4着以下)
    
    model_win = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42, n_jobs=-1)
    model_win.fit(X_win, y_win)
    
    win_model_file = os.path.join(WORK_DIR, f'keirin_win_model_{current_ym}.pkl')
    joblib.dump(model_win, win_model_file)
    print(f"   ✅ 勝率予測AIモデルを保存しました: {os.path.basename(win_model_file)}")

    # =====================================================================
    # 5. オッズ予測AI（万車券判定AI）の再学習と保存
    # =====================================================================
    print("\n>> 🤖 オッズ予測AIモデル（大衆心理）を最新データで再学習中...")
    
    # groupby-applyのインデックスバグを回避するためtransformを使用
    golden_df['kd_score'] = golden_df.groupby('race_id')['score'].transform(lambda x: (x**5)/(x**5).sum())
    
    # 同着による重複エラーを防ぐため drop_duplicates を追加
    df_1 = golden_df[golden_df['target']==0].drop_duplicates('race_id').set_index('race_id')[['score', 'b_count', 'payout_yen']].rename(columns={'score':'s1', 'b_count':'b1'})
    df_2 = golden_df[golden_df['target']==1].drop_duplicates('race_id').set_index('race_id')[['score', 'b_count']].rename(columns={'score':'s2', 'b_count':'b2'})
    df_3 = golden_df[golden_df['target']==2].drop_duplicates('race_id').set_index('race_id')[['score', 'b_count']].rename(columns={'score':'s3', 'b_count':'b3'})
    
    df_kd1 = golden_df[golden_df['target']==0].drop_duplicates('race_id').set_index('race_id')[['kd_score']].rename(columns={'kd_score':'kd1'})
    df_kd2 = golden_df[golden_df['target']==1].drop_duplicates('race_id').set_index('race_id')[['kd_score']].rename(columns={'kd_score':'kd2'})
    df_kd3 = golden_df[golden_df['target']==2].drop_duplicates('race_id').set_index('race_id')[['kd_score']].rename(columns={'kd_score':'kd3'})
    
    odds_df = df_1.join(df_2).join(df_3).join(df_kd1).join(df_kd2).join(df_kd3).dropna()
    
    # 擬似オッズ計算
    den2 = (1.0 - odds_df['kd1']).clip(lower=1e-6)
    den3 = (1.0 - odds_df['kd1'] - odds_df['kd2']).clip(lower=1e-6)
    pub_prob = odds_df['kd1'] * (odds_df['kd2'] / den2) * (odds_df['kd3'] / den3)
    odds_df['pseudo_odds'] = (0.75 / pub_prob).clip(upper=9999.0)
    
    odds_df['s_diff12'] = odds_df['s1'] - odds_df['s2']
    odds_df['s_diff13'] = odds_df['s1'] - odds_df['s3']
    odds_df['is_manshaken'] = (odds_df['payout_yen'] >= 10000).astype(int)
    
    features_odds = ['s1', 's2', 's3', 'b1', 'b2', 'b3', 'pseudo_odds', 's_diff12', 's_diff13']
    
    model_odds = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
    model_odds.fit(odds_df[features_odds], odds_df['is_manshaken'])
    
    odds_model_file = os.path.join(WORK_DIR, f'keirin_odds_model_{current_ym}.pkl')
    joblib.dump(model_odds, odds_model_file)
    print(f"   ✅ オッズ予測AIモデルを保存しました: {os.path.basename(odds_model_file)}")

    print("\n" + "="*60)
    print("🎉 月次アップデートがすべて正常に完了しました！")
    print(f"作成された最新モデル群（{current_ym}版）は、明日からの日次スクリプトで自動的に読み込まれます。")
    print("="*60)

if __name__ == "__main__":
    run_monthly_training_pipeline()


