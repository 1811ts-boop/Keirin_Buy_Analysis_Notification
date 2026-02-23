import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re
from datetime import datetime, timedelta
import concurrent.futures
from tqdm import tqdm
import os
import glob
import itertools
import joblib
import warnings

# 🌟 修正必須1対応済：Colabのタイムゾーンを日本時間(JST)に強制設定
os.environ['TZ'] = 'Asia/Tokyo'
time.tzset()

warnings.filterwarnings('ignore')

# 🌟Googleドライブのマウント処理
# from google.colab import drive
# drive.mount('/content/drive')

# ==========================================
# ⚙️ 設定クラス (本番稼働用)
# ==========================================
class Config:
    DRIVE_DIR = './KeirinData'
    TOMORROW_FILE = 'tomorrow_races.csv'
    MAX_WORKERS = 1 
    SLEEP_TIME = 1.0
    
    # 🌟 変更：Messaging API用の2つの変数を読み込む
    LINE_CHANNEL_TOKEN = os.environ.get('LINE_CHANNEL_TOKEN', 'YOUR_TOKEN')
    LINE_USER_ID = os.environ.get('LINE_USER_ID', 'YOUR_USER_ID')

# ==========================================
# フェーズ1：スクレイピング・コアロジック
# ==========================================
BANK_MAP = {
    '22': '335', '31': '333', '36': '333', '37': '333', '46': '333', '53': '333', '63': '333',
    '25': '500', '26': '500', '74': '500'
}

PREFECTURES = [
    "北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島", "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
    "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜", "静岡", "愛知", "三重", "滋賀", "京都", "大阪", "兵庫",
    "奈良", "和歌山", "鳥取", "島根", "岡山", "広島", "山口", "徳島", "香川", "愛媛", "高知", "福岡", "佐賀", "長崎",
    "熊本", "大分", "宮崎", "鹿児島", "沖縄"
]

class KDreamsAnalysisScraper:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        self.base_url = "https://keirin.kdreams.jp"

    def get_soup(self, url):
        try:
            time.sleep(Config.SLEEP_TIME)
            if url.startswith("/"): url = self.base_url + url
            res = self.session.get(url, timeout=15)
            res.encoding = res.apparent_encoding
            if res.status_code == 200: return BeautifulSoup(res.text, 'html.parser')
            return None
        except: return None

    def clean_text(self, text):
        return re.sub(r'\s+', ' ', text).strip() if text else "-"

    def fetch_race_urls_daily(self, date_obj):
        date_str = date_obj.strftime('%Y/%m/%d')
        target_date_id = date_obj.strftime('%Y%m%d')
        url = f"{self.base_url}/kaisai/{date_str}/"
        soup = None
        for _ in range(3):
            soup = self.get_soup(url)
            if soup: break
            time.sleep(3) 
        if not soup: return None
        unique_urls = set()
        for link in soup.find_all('a', href=re.compile(r'/racedetail/')):
            href = link.get('href').split('?')[0]
            if target_date_id in href: unique_urls.add(href)
        return list(unique_urls)

    def _extract_race_info(self, soup, url, date_str):
        meta = {'date': date_str, 'url': url}
        try:
            title_full = self.clean_text(soup.title.text)
            meta['race_title_full'] = title_full
            meta['place_name'] = title_full.split(' ')[0].replace('競輪', '')
            match = re.search(r'/racedetail/(\d{2})(\d{8})(\d{2})', url)
            if match: meta['place_code'], _, meta['race_num'] = match.groups()
            else: meta['place_code'] = meta['race_num'] = "-"
            bank_len_str = BANK_MAP.get(meta['place_code'], '400')
            meta['bank_length'] = bank_len_str
            meta['round'] = "一般"
            for r in ["予選", "特選", "決勝", "準決勝", "選抜"]:
                if r in title_full: meta['round'] = r; break
            info_text = soup.text
            dist_match = re.search(r'(\d{1,2},?\d{3})\s*m', info_text)
            if dist_match: meta['distance'] = dist_match.group(1).replace(',', '')
            else: meta['distance'] = "-" 
            time_match = re.search(r'(?:^|\s|発走)(\d{1,2}:\d{2})(?:$|\s|発走)', info_text)
            meta['start_time'] = time_match.group(1) if time_match else "-"
            is_midnight = "ミッドナイト" in title_full
            if not is_midnight and meta['start_time'] != "-":
                try:
                    h = int(meta['start_time'].split(':')[0])
                    m = int(meta['start_time'].split(':')[1])
                    if h >= 21 or (h == 20 and m >= 30): is_midnight = True
                except: pass
            is_girls = "ガールズ" in title_full or "L級" in title_full
            meta['race_type'] = "P3" if is_girls else ("P1" if is_midnight else "P2")
            meta['race_type_detail'] = meta['race_type']
            if meta['race_type'] == "P2":
                if any(x in title_full for x in ["S級", "Ｓ級"]): meta['race_type_detail'] = "P2-S"
                elif any(x in title_full for x in ["チャレンジ"]): meta['race_type_detail'] = "P2-Chal"
                elif any(x in title_full for x in ["A級", "Ａ級", "特選", "選抜", "予選"]): meta['race_type_detail'] = "P2-A12"
            if meta['distance'] == "-":
                dist = "1625"
                bl = int(bank_len_str)
                if bl == 333 or bl == 335: dist = "1662"
                elif bl == 400:
                    if meta['race_type'] == "P3" or meta['race_type_detail'] == "P2-Chal" or meta['race_type_detail'] == "P1": dist = "1625"
                    else: dist = "2025"
                elif bl == 500:
                    if meta['race_type'] == "P3": dist = "1525"
                    else: dist = "2025"
                meta['distance'] = dist
            meta['line_prediction'] = "-"
            dt = soup.find('dt', string=re.compile('並び予想'))
            if dt:
                dd = dt.find_next_sibling('dd')
                if dd:
                    parts = []
                    spans = dd.find_all('span', class_='icon_p')
                    for sp in spans:
                        classes = sp.get('class', [])
                        if 'space' in classes: parts.append("|")
                        else:
                            txt = self.clean_text(sp.text)
                            if txt and txt != "←": parts.append(txt)
                    if parts:
                        full_line = "← " + " ".join(parts)
                        full_line = re.sub(r'(\s*\|\s*)+', ' | ', full_line)
                        full_line = full_line.strip().rstrip('|').strip()
                        meta['line_prediction'] = full_line
        except: return None
        return meta

    def _parse_name_cell(self, raw_text):
        text = re.sub(r'\s+', ' ', raw_text).strip()
        age, grad = "", ""
        match = re.search(r'/(\d{1,3})/([A-Z0-9]+)$', text)
        remaining = text
        if match:
            age, grad = match.group(1), match.group(2)
            remaining = text[:match.start()].strip()
        clean_remaining = remaining.replace(" ", "")
        area, name = "", remaining
        for pref in PREFECTURES:
            if clean_remaining.endswith(pref):
                area = pref
                chars_to_remove = list(pref)
                temp_text = list(remaining)
                while chars_to_remove and temp_text:
                    if temp_text[-1] == chars_to_remove[-1]:
                        chars_to_remove.pop()
                        temp_text.pop()
                    elif temp_text[-1] == ' ': temp_text.pop()
                    else: break
                if not chars_to_remove: name = "".join(temp_text).strip()
                break
        return name, area, age, grad

    def _extract_players(self, soup):
        players = {}
        target_table = None
        for t in soup.find_all('table'):
            if "競走得点" in t.text: target_table = t; break
        if target_table:
            rows = target_table.find_all('tr')
            for row in rows:
                cells = [self.clean_text(c.text) for c in row.find_all(['td','th'])]
                idx_offset, car_num = -1, "-"
                for i in range(len(cells) - 1):
                    if cells[i].isdigit() and 1 <= int(cells[i]) <= 9:
                        if not cells[i+1].isdigit():
                            car_num, idx_offset = cells[i], i
                            break
                if idx_offset == -1: continue
                def safe_get(rel_idx): return cells[idx_offset + rel_idx] if idx_offset + rel_idx < len(cells) else "-"
                name, area, age, grad = self._parse_name_cell(safe_get(1))
                players[car_num] = {
                    'name': name, 'area': area, 'age': age, 'grad': grad,
                    'class': safe_get(2), 'leg': safe_get(3), 'gear': safe_get(4),
                    'score': safe_get(5), 's': safe_get(6), 'b': safe_get(7),
                    'win': safe_get(16), 'ren2': safe_get(17), 'ren3': safe_get(18)
                }
        return players

    def _extract_results(self, base_url):
        result_url = base_url + "?pageType=showResult"
        soup = self.get_soup(result_url)
        payout_res, ranks = {'payout_yen': 0, 'payout_pop': 0}, {}
        if not soup: return payout_res, ranks
        payout_text = ""
        try:
            for dt in soup.find_all('dt'):
                txt = dt.text
                if "3連単" in txt or re.match(r'\d+-\d+-\d+', txt):
                    dd = dt.find_next_sibling('dd')
                    if dd: payout_text = self.clean_text(dd.text)
        except: pass
        if payout_text:
            match = re.search(r'(\d+)\((\d+)\)', payout_text.replace(',', '').replace('円', ''))
            if match:
                payout_res['payout_yen'], payout_res['payout_pop'] = int(match.group(1)), int(match.group(2))
        res_table = soup.find('table', class_=re.compile('result_table')) or soup.find('table')
        if res_table:
            for row in res_table.find_all('tr'):
                cells = [self.clean_text(c.text) for c in row.find_all('td')]
                rank_val, target_car, rank_idx = "-", "-", -1
                for i, cell in enumerate(cells):
                    if rank_val == "-":
                        m = re.search(r'^(\d+)$', cell)
                        if m and 1 <= int(m.group(1)) <= 9: rank_val, rank_idx = m.group(1), i
                        elif cell in ['失','欠','落','故']: rank_val, rank_idx = cell, i
                    m_car = re.search(r'^(\d+)$', cell)
                    if m_car and i != rank_idx: target_car = m_car.group(1)
                if target_car != "-" and rank_val != "-": ranks[target_car] = rank_val
        return payout_res, ranks

    def parse_one_race(self, url, date_str):
        soup_entry = self.get_soup(url)
        if not soup_entry: return None
        meta = self._extract_race_info(soup_entry, url, date_str)
        if not meta: return None
        players = self._extract_players(soup_entry)
        payouts, ranks = self._extract_results(url)
        row = meta.copy()
        row.update(payouts)
        row['car_count'] = len(players)
        for i in range(1, 8):
            prefix, car_key = f"c{i}", str(i)
            exists = car_key in players
            row[f"{prefix}_existence"] = 1 if exists else 0
            if exists:
                p = players[car_key]
                row[f"{prefix}_name"] = p['name']
                row[f"{prefix}_area"] = p['area']
                row[f"{prefix}_age"] = p['age']
                row[f"{prefix}_grad"] = p['grad']
                row[f"{prefix}_class"] = p['class']
                row[f"{prefix}_leg"] = p['leg']
                row[f"{prefix}_score"] = p['score']
                row[f"{prefix}_win"] = p['win']
                row[f"{prefix}_2ren"] = p['ren2']
                row[f"{prefix}_b"] = p['b']
                row[f"{prefix}_rank"] = ranks.get(car_key, "0")
            else:
                for k in ['name', 'area', 'age', 'grad', 'class', 'leg', 'score', 'win', '2ren', 'b', 'rank']:
                    row[f"{prefix}_{k}"] = 0 if k in ['score', 'win', '2ren', 'b', 'rank'] else ""
        return row

def get_latest_file(pattern):
    files = glob.glob(os.path.join(Config.DRIVE_DIR, pattern))
    if not files: return None
    return sorted(files)[-1]

def send_line_notify(message):
    """🌟 変更：LINE Messaging APIを使ってプッシュメッセージを送信する"""
    token = Config.LINE_CHANNEL_TOKEN
    user_id = Config.LINE_USER_ID
    
    if token == 'YOUR_TOKEN' or user_id == 'YOUR_USER_ID':
        print("LINEトークンまたはユーザーIDが設定されていないため、通知をスキップします。")
        return
        
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    data = {
        "to": user_id,
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }
    
    # APIへ送信
    res = requests.post(url, headers=headers, json=data)
    if res.status_code != 200:
        print(f"❌ LINE送信エラー: [{res.status_code}] {res.text}")

# ==========================================
# フェーズ2：AI用特徴量エンジニアリング (架け橋ロジック)
# ==========================================
def generate_features_for_inference(df_today):
    records = []
    for idx, row in df_today.iterrows():
        race_id = f"{row['date']}_{row['place_name']}_{row['race_num']}"
        line_str = str(row['line_prediction'])
        lines = []
        if line_str and line_str != "-":
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
                players[str(i)] = {
                    'score': score, 
                    'b_count': b_count, 
                    'leg': str(row.get(f"{prefix}_leg", "")), 
                    'area': str(row.get(f"{prefix}_area", ""))
                }
                
        if not players: continue
        max_score = max([p['score'] for p in players.values()])
        chaos_idx = np.std([p['score'] for p in players.values()])
        
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
                'race_id': race_id, 'date': row['date'], 'place_name': row['place_name'], 'race_num': row['race_num'],
                'car_number': int(car_num), 'score': p_info['score'], 'b_count': p_info['b_count'],
                'score_diff_from_max': max_score - p_info['score'], 'position_in_line': position_in_line,
                'line_length': line_length, 'is_solo': is_solo, 'leader_score': leader_score,
                'leader_b_count': leader_b_count, 'is_same_area_as_leader': is_same_area,
                'chaos_idx': chaos_idx, 'bank_length': str(row.get('bank_length', '400')), 'leg': p_info['leg'],
                'total_lines': total_lines
            })
    return pd.DataFrame(records)

# ==========================================
# フェーズ3：推論とLINEスナイプ指令ロジック
# ==========================================
def run_ai_sniper(df_features):
    today_str = datetime.now().strftime('%Y-%m-%d')
    print("\n=== 🎯 Step 2: AIスナイプ推論 ===")
    
    threshold_file = get_latest_file('keirin_threshold_*.pkl')
    win_model_file = get_latest_file('keirin_win_model_*.pkl')
    odds_model_file = get_latest_file('keirin_odds_model_*.pkl')
    
    if not all([threshold_file, win_model_file, odds_model_file]):
        print("❌ 必要なAIモデルファイルが見つかりません。")
        return
        
    thresholds = joblib.load(threshold_file)
    model_win = joblib.load(win_model_file)
    model_odds = joblib.load(odds_model_file)
    
    race_stats = df_features.groupby('race_id')['score'].std().reset_index(name='score_std')
    df_features['score_gap_type'] = pd.cut(df_features['race_id'].map(race_stats.set_index('race_id')['score_std']), bins=thresholds['score_bins'], labels=['拮抗(小)', '普通(中)', '鉄板(大)'])
    b_stats = df_features.groupby('race_id')['b_count'].sum().reset_index(name='b_sum')
    df_features['b_sum_type'] = pd.cut(df_features['race_id'].map(b_stats.set_index('race_id')['b_sum']), bins=thresholds['b_bins'], labels=['先行不在(少)', '標準(中)', 'モガキ合い(多)'])
    
    golden_df = df_features[
        (df_features['score_gap_type'] == '拮抗(小)') & 
        (df_features['b_sum_type'] == 'モガキ合い(多)') & 
        (df_features['total_lines'] >= 4)
    ].copy()
    
    if golden_df.empty:
        send_line_notify(f"\n📅 {today_str}\n本日は『黄金の乱戦』を満たすレースが一つもありませんでした。")
        return

    features_win = ['score', 'b_count', 'score_diff_from_max', 'position_in_line', 'line_length', 'is_solo', 'leader_score', 'leader_b_count', 'is_same_area_as_leader', 'chaos_idx', 'bank_length', 'leg']
    for col in ['bank_length', 'leg']: golden_df[col] = golden_df[col].astype('category')
    probs = model_win.predict_proba(golden_df[features_win])
    golden_df['prob_1st'], golden_df['prob_2nd'], golden_df['prob_3rd'] = probs[:, 0], probs[:, 1], probs[:, 2]
    
# === 差し替えここから ===
    message_lines = [f"📅 {today_str} 大穴スナイプ指令\n"]
    race_count = 0
    circle_nums = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
    
    for race_id, group in golden_df.groupby('race_id'):
        if len(group) != 7: continue 
        
        # 確率・オッズ・トップ10抽出の計算 (変更なし)
        p1, p2, p3 = dict(zip(group['car_number'], group['prob_1st'])), dict(zip(group['car_number'], group['prob_2nd'])), dict(zip(group['car_number'], group['prob_3rd']))
        s_dict, b_dict = dict(zip(group['car_number'], group['score'])), dict(zip(group['car_number'], group['b_count']))
        kd_dict = dict(zip(group['car_number'], (group['score']**5) / (group['score']**5).sum()))
        car_numbers = group['car_number'].tolist()
        
        all_bets = []
        for c1, c2, c3 in itertools.permutations(car_numbers, 3):
            den2, den3 = max(1.0 - p2[c1], 1e-6), max(1.0 - p3[c1] - p3[c2], 1e-6)
            ai_prob = p1[c1] * (p2[c2] / den2) * (p3[c3] / den3)
            
            den2_kd, den3_kd = max(1.0 - kd_dict[c1], 1e-6), max(1.0 - kd_dict[c1] - kd_dict[c2], 1e-6)
            pub_prob = kd_dict[c1] * (kd_dict[c2] / den2_kd) * (kd_dict[c3] / den3_kd)
            pseudo_odds = 0.75 / pub_prob if pub_prob > 0 else 9999.0
            
            s1, s2, s3 = s_dict[c1], s_dict[c2], s_dict[c3]
            b1, b2, b3 = b_dict[c1], b_dict[c2], b_dict[c3]
            all_bets.append({'combo': f"{c1}-{c2}-{c3}", 'prob': ai_prob, 'x_odds': [s1, s2, s3, b1, b2, b3, pseudo_odds, s1-s2, s1-s3]})
            
        all_bets.sort(key=lambda b: b['prob'], reverse=True)
        top_10 = all_bets[:10]
        manshaken_probs = model_odds.predict_proba([b['x_odds'] for b in top_10])[:, 1]
        
        # 🌟 フォーマット変更：見出しとカラムヘッダー
        message_lines.append(f"🏁 【{group['place_name'].iloc[0]} {group['race_num'].iloc[0]}R】")
        message_lines.append("買い目｜勝率｜オッズ100％期待度（🔥🟡:買｜💧:見送）")
        
        for i, b in enumerate(top_10):
            exp_rate = manshaken_probs[i] * 100
            if exp_rate >= 50.0: mark = "🔥" 
            elif exp_rate >= 30.0: mark = "🟡" 
            else: mark = "💧" 
            
            # 🌟 丸数字を使ってコンパクトにまとめる
            rank_mark = circle_nums[i] if i < 10 else f"{i+1}."
            message_lines.append(f"{rank_mark}{b['combo']}｜{b['prob']*100:.1f}%｜{mark}{exp_rate:.1f}%")
            
        message_lines.append("-------------------------")
        race_count += 1
        
        # 🌟 3レースごとに分割送信する「バッチ処理」
        if race_count % 3 == 0:
            send_line_notify("\n".join(message_lines))
            message_lines = [f"📅 {today_str} 大穴スナイプ指令 (続き)\n"] # バッファをリセット
            time.sleep(1) # API制限回避のウェイト
            
    # 🌟 ループ終了後、送信しきれていない残り（1〜2レース分）があれば送信
    if len(message_lines) > 1: # タイトル行以外に中身があれば
        send_line_notify("\n".join(message_lines))

    print(">> ✅ AIスナイプ指令をLINEに送信完了！")
    # === 差し替えここまで ===
# ==========================================
# 🚀 実行メインブロック
# ==========================================
def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 日次自動バッチ処理開始")


    os.makedirs(Config.DRIVE_DIR, exist_ok=True)
    
    full_path_master = get_latest_file('kdreams_analysis_*_master.csv')
    full_path_tomorrow = os.path.join(Config.DRIVE_DIR, Config.TOMORROW_FILE)
    
    start_dt = None
    if full_path_master:
        try:
            df_temp = pd.read_csv(full_path_master, usecols=['date'], low_memory=False)
            max_date = pd.to_datetime(df_temp['date'], errors='coerce').max()
            if pd.notna(max_date):
                start_dt = max_date + timedelta(days=1)
                print(f"📊 マスターCSV最新日付: {max_date.strftime('%Y-%m-%d')} ({os.path.basename(full_path_master)})")
        except Exception as e: print(f"⚠️ マスター読込エラー: {e}")
            
    if start_dt is None:
        start_dt = datetime.now() - timedelta(days=1)
        print(f"⚠️ 最新日付が取得不可のため、{start_dt.strftime('%Y-%m-%d')}から開始します。")

    if not full_path_master:
        full_path_master = os.path.join(Config.DRIVE_DIR, f"kdreams_analysis_{start_dt.strftime('%Y%m%d')}_master.csv")

    today_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = today_dt

    if start_dt <= end_dt:
        days_total = (end_dt - start_dt).days + 1
        date_list = [start_dt + timedelta(days=x) for x in range(days_total)]
        scraper = KDreamsAnalysisScraper()
        
        first_cols = ['date', 'place_name', 'race_num', 'race_type_detail', 'race_title_full', 'bank_length', 'distance', 'start_time', 'payout_yen', 'payout_pop', 'line_prediction']
        
        print(f"\n=== 🚀 Step 1: データ自動収集＆仕分け ({start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}) ===")
        for target_date in tqdm(date_list, desc="Scraping Progress"):
            urls = scraper.fetch_race_urls_daily(target_date)
            if not urls: continue
                
            daily_data = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                futures = [executor.submit(scraper.parse_one_race, u, target_date.strftime('%Y-%m-%d')) for u in urls]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res: daily_data.append(res)
            
            if not daily_data: continue
            df_day = pd.DataFrame(daily_data)
            final_cols = [c for c in first_cols if c in df_day.columns] + [c for c in df_day.columns if c not in first_cols]
            df_day = df_day[final_cols]

            if target_date == today_dt:
                df_day.to_csv(full_path_tomorrow, mode='w', header=True, index=False, encoding='utf-8-sig')
                print(f"📝 {target_date.strftime('%Y-%m-%d')} の出走表をテスト用紙に保存しました。")
            else:
                file_exists = os.path.exists(full_path_master)
                if not file_exists:
                    df_day.to_csv(full_path_master, mode='w', header=True, index=False, encoding='utf-8-sig')
                    print(f"🌟 新規マスターファイルを作成し、{target_date.strftime('%Y-%m-%d')} の結果を保存しました。")
                else:
                    df_day.to_csv(full_path_master, mode='a', header=False, index=False, encoding='utf-8-sig')
                    print(f"💾 {target_date.strftime('%Y-%m-%d')} の結果をマスターに追記しました。")
            time.sleep(2)
            
    # ==========================================
    # 🌟 フェーズ2 & 3 の実行 (修正フィルター適用済)
    # ==========================================
    if os.path.exists(full_path_tomorrow):
        df_today = pd.read_csv(full_path_tomorrow)
        
        today_str = datetime.now().strftime('%Y-%m-%d')
        df_today = df_today[df_today['date'] == today_str]
        
        if not df_today.empty:
            df_features = generate_features_for_inference(df_today)
            if not df_features.empty:
                run_ai_sniper(df_features)
            else:
                print(f"[{today_str}] 推論可能な選手データが抽出できなかったためスキップします。")
        else:
            print(f"[{today_str}] 本日の新しい出走表データが存在しないため、AI推論をスキップします。")
    
    print("\n=== ✅ 日次バッチ処理 全工程完了 ===")

if __name__ == "__main__":
    main()


