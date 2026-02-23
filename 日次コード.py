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

# ==========================================
# 🌟 環境設定 (GitHub Actions 本番用)
# ==========================================
os.environ['TZ'] = 'Asia/Tokyo'
try:
    time.tzset()
except:
    pass
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ 設定クラス (環境変数からLINE APIキーを取得)
# ==========================================
class Config:
    DRIVE_DIR = './KeirinData' 
    TOMORROW_FILE = 'today_races.csv'
    MAX_WORKERS = 1 
    SLEEP_TIME = 1.0
    
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
        url = f"{self.base_url}/kaisai/{date_str}//"
        
        soup = None
        for _ in range(3):
            soup = self.get_soup(url)
            if soup: break
            time.sleep(3) 
        if not soup: return None
        
        racecard_urls = set()
        for li in soup.find_all('li'):
            if 'active' in li.get('class', []):
                for link in li.find_all('a', href=re.compile(r'/racecard/\d+')):
                    href = link.get('href').split('?')[0]
                    if not href.startswith("http"):
                        href = self.base_url + href
                    racecard_urls.add(href)

        all_race_urls = set()
        for rc_url in racecard_urls:
            match = re.search(r'/racecard/(\d+)/?', rc_url)
            if not match: continue
            base_id = match.group(1)
            
            rc_soup = self.get_soup(rc_url)
            if rc_soup:
                for link in rc_soup.find_all('a', href=re.compile(f'/racedetail/{base_id}')):
                    href = link.get('href').split('?')[0]
                    if not href.startswith("http"):
                        href = self.base_url + href
                    all_race_urls.add(href)
                        
        return list(all_race_urls)

    def _extract_race_info(self, soup, url, date_str):
        meta = {'date': date_str, 'url': url}
        try:
            title_full = self.clean_text(soup.title.text)
            meta['race_title_full'] = title_full
            meta['place_name'] = title_full.split(' ')[0].replace('競輪', '')
            match = re.search(r'/racedetail/(\d{2})(\d{8}).*?(\d{2})/?$', url)
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
        
        players = self._extract_players(soup_entry)
        if not players: return None
        max_car = max([int(k) for k in players.keys()])
        if max_car > 7:
            return None 

        meta = self._extract_race_info(soup_entry, url, date_str)
        if not meta: return None
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

# ==========================================
# 本番用：LINE通知送信関数
# ==========================================
def send_line_notify(message):
    if Config.LINE_CHANNEL_TOKEN in ['YOUR_TOKEN', 'TEST_TOKEN']:
        print("📱 【LINE通知シミュレーション（環境変数が未設定です）】\n" + message)
        return

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.LINE_CHANNEL_TOKEN}"
    }
    data = {
        "to": Config.LINE_USER_ID,
        "messages": [{"type": "text", "text": message}]
    }
    try:
        res = requests.post(url, headers=headers, json=data)
        if res.status_code != 200:
            print(f"❌ LINE通知エラー: {res.text}")
    except Exception as e:
        print(f"❌ LINE通信エラー: {e}")

# ==========================================
# フェーズ2：AI用特徴量エンジニアリング
# ==========================================
def generate_features_for_inference(df_today):
    records = []
    for idx, row in df_today.iterrows():
        race_id = f"{row['date']}_{row['place_name']}_{row['race_num']}"
        
        line_str = str(row['line_prediction'])
        lines = []
        if pd.notna(line_str) and line_str != '-' and line_str != '':
            line_str = line_str.replace('←', '')
            line_str = line_str.replace('｜', '-').replace('|', '-')
            line_str = line_str.replace('/', '-').replace(' ', '-').replace('　', '-')
            line_str = re.sub(r'[^0-9\-]', '', line_str)
            line_str = re.sub(r'\-+', '-', line_str).strip('-')
            
            if line_str:
                for line_group in line_str.split('-'):
                    if line_group:
                        lines.append(list(line_group))
        
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
            
            try: p_code = int(row.get('place_code', 99))
            except: p_code = 99
            
            records.append({
                'race_id': race_id, 'date': row['date'], 'place_name': row['place_name'], 
                'race_num': int(row['race_num']), 'place_code': p_code,
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
def run_ai_sniper(df_features, today_str):
    print(f"\n=== Step 2: AIスナイプ推論 ({today_str}) ===")
    
    df_features = df_features.sort_values(by=['place_code', 'race_num'])
    
    threshold_file = get_latest_file('keirin_thresholds_*.pkl')
    win_model_file = get_latest_file('keirin_win_model_*.pkl')
    odds_model_file = get_latest_file('keirin_odds_model_*.pkl')
    
    if not (threshold_file and win_model_file and odds_model_file):
        print("❌ 必要なAIモデルファイルが見つかりません。")
        return
        
    thresholds = joblib.load(threshold_file)
    model_win = joblib.load(win_model_file)
    model_odds = joblib.load(odds_model_file)
    
    race_stats = df_features.groupby('race_id', sort=False)['score'].std().reset_index(name='score_std')
    b_stats = df_features.groupby('race_id', sort=False)['b_count'].sum().reset_index(name='b_sum')
    
    df_features['score_gap_type'] = pd.cut(df_features['race_id'].map(race_stats.set_index('race_id')['score_std']), bins=thresholds['score_bins'], labels=['拮抗(小)', '普通(中)', '鉄板(大)'])
    df_features['b_sum_type'] = pd.cut(df_features['race_id'].map(b_stats.set_index('race_id')['b_sum']), bins=thresholds['b_bins'], labels=['先行不在(少)', '標準(中)', 'モガキ合い(多)'])

    print(f"\n=== 🔍 {today_str} 全レース『黄金の乱戦』判定レポート ===")
    golden_count = 0
    for race_id, group in df_features.groupby('race_id', sort=False):
        score_gap = group['score_gap_type'].iloc[0]
        b_sum = group['b_sum_type'].iloc[0]
        total_lines = group['total_lines'].iloc[0]
        
        is_golden = (score_gap == '拮抗(小)') and (b_sum == 'モガキ合い(多)') and (total_lines >= 4)
        
        if is_golden:
            print(f"✅ 【合格】 {race_id}: (実力差:{score_gap}, B本数:{b_sum}, ライン:{total_lines}本)")
            golden_count += 1
        else:
            reasons = []
            if score_gap != '拮抗(小)': reasons.append(f"実力差が{score_gap}")
            if b_sum != 'モガキ合い(多)': reasons.append(f"B本数が{b_sum}")
            if total_lines < 4: reasons.append(f"ライン数が{total_lines}本")
            print(f"❌ 【除外】 {race_id}: 理由 -> {', '.join(reasons)}")
            
    print(f"--------------------------------------------------")
    print(f"📊 本日の判定結果: 全 {df_features['race_id'].nunique()} レース中、条件クリアは {golden_count} レース")
    print("==================================================\n")

    golden_df = df_features[(df_features['score_gap_type'] == '拮抗(小)') & (df_features['b_sum_type'] == 'モガキ合い(多)') & (df_features['total_lines'] >= 4)].copy()

    if golden_df.empty:
        send_line_notify(f"\n ***{today_str}***\n本日は『黄金の乱戦』を満たすレースが一つもありませんでした。")
        print("本日は条件合致レースなし。")
        return

    features_win = ['score', 'b_count', 'score_diff_from_max', 'position_in_line', 'line_length', 'is_solo', 'leader_score', 'leader_b_count', 'is_same_area_as_leader', 'chaos_idx', 'bank_length', 'leg']
    for col in ['bank_length', 'leg']: golden_df[col] = golden_df[col].astype('category')
    probs = model_win.predict_proba(golden_df[features_win])
    golden_df['prob_1st'], golden_df['prob_2nd'], golden_df['prob_3rd'] = probs[:, 0], probs[:, 1], probs[:, 2]
    
    message_lines = [f"📅 {today_str} 大穴スナイプ指令\n"]
    race_count = 0
    circle_nums = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
    
    for race_id, group in golden_df.groupby('race_id', sort=False):
        if len(group) != 7: continue 
        
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
        
        message_lines.append(f"🏁 【{group['place_name'].iloc[0]} {group['race_num'].iloc[0]}R】")
        message_lines.append("買い目｜勝率｜オッズ100％期待度（🔥🟡:買｜💧:見送）")
        
        for i, b in enumerate(top_10):
            exp_rate = manshaken_probs[i] * 100
            if exp_rate >= 50.0: mark = "🔥" 
            elif exp_rate >= 30.0: mark = "🟡" 
            else: mark = "💧" 
            
            rank_mark = circle_nums[i] if i < 10 else f"{i+1}."
            message_lines.append(f"{rank_mark}{b['combo']}｜{b['prob']*100:.1f}%｜{mark}{exp_rate:.1f}%")
            
        message_lines.append("-------------------------")
        race_count += 1
        
        if race_count % 3 == 0:
            send_line_notify("\n".join(message_lines))
            message_lines = [f"📅 {today_str} 大穴スナイプ指令 (続き)\n"]
        
    if len(message_lines) > 1:
        send_line_notify("\n".join(message_lines))

    print(">> ✅ AIスナイプ指令の生成完了！")

# ==========================================
# 🚀 実行メインブロック（本番：毎朝4時バッチ用）
# ==========================================
def main():
    # 毎日朝4時の実行を想定。昨日をstart_dt（マスター追記用）、今日をend_dt（スナイプ用）に設定
    today_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = today_dt - timedelta(days=1)
    end_dt = today_dt
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 日次自動バッチ処理開始 (本番環境)")

    os.makedirs(Config.DRIVE_DIR, exist_ok=True)
    full_path_master = os.path.join(Config.DRIVE_DIR, 'keirin_master.csv')
    full_path_tomorrow = os.path.join(Config.DRIVE_DIR, Config.TOMORROW_FILE)

    if start_dt <= end_dt:
        days_total = (end_dt - start_dt).days + 1
        date_list = [start_dt + timedelta(days=x) for x in range(days_total)]
        scraper = KDreamsAnalysisScraper()
        
        first_cols = ['date', 'place_name', 'race_num', 'race_type_detail', 'race_title_full', 'bank_length', 'distance', 'start_time', 'payout_yen', 'payout_pop', 'line_prediction']
        
        print(f"\n=== 🚀 Step 1: データ自動収集＆仕分け ===")
        for target_date in date_list:
            target_date_str = target_date.strftime('%Y-%m-%d')
            print(f"\n{'='*55}")
            print(f"📅 【 {target_date_str} 】の処理を開始します")
            print(f"{'='*55}")
            
            urls = scraper.fetch_race_urls_daily(target_date)
            if not urls: 
                print(f"❌ {target_date_str} のレースURLが取得できませんでした。")
                continue
            
            print(f"🔍 取得したレースURL数: {len(urls)} 件")
                
            daily_data = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                futures = [executor.submit(scraper.parse_one_race, u, target_date_str) for u in urls]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res: daily_data.append(res)
            
            if not daily_data: 
                continue
            
            df_day = pd.DataFrame(daily_data)
            final_cols = [c for c in first_cols if c in df_day.columns] + [c for c in df_day.columns if c not in first_cols]
            df_day = df_day[final_cols]

            # ----------------------------------------------------
            # 昨日のデータはマスター(keirin_master.csv)に追記、
            # 今日のデータは推論用(today_races.csv)に上書き保存
            # ----------------------------------------------------
            if target_date == today_dt:
                df_day.to_csv(full_path_tomorrow, mode='w', header=True, index=False, encoding='utf-8-sig')
                print(f"📝 {target_date_str} の出走表をテスト用紙(today_races.csv)に保存しました。")
            else:
                file_exists = os.path.exists(full_path_master)
                if not file_exists:
                    df_day.to_csv(full_path_master, mode='w', header=True, index=False, encoding='utf-8-sig')
                    print(f"🌟 新規マスターファイルを作成し、保存しました。")
                else:
                    try:
                        master_cols = pd.read_csv(full_path_master, nrows=0, low_memory=False).columns.tolist()
                        for col in master_cols:
                            if col not in df_day.columns:
                                df_day[col] = np.nan
                        df_day_aligned = df_day[master_cols]
                        df_day_aligned.to_csv(full_path_master, mode='a', header=False, index=False, encoding='utf-8-sig')
                        print(f"💾 {target_date_str} の結果（着順・配当入り）をマスターデータに安全に追記しました。")
                    except Exception as e:
                        print(f"❌ マスター追記時にエラー発生: {e}")
            
            time.sleep(2)
            
            # ----------------------------------------------------
            # 今日のデータ（出走表）に対してのみAI推論を実行する
            # ----------------------------------------------------
            if target_date == today_dt:
                df_today = df_day.copy()
                
                # 鉄壁フィルター1: 8番車が存在するレースを物理的に排除
                if 'c8_existence' in df_today.columns:
                    df_today = df_today[df_today['c8_existence'] != 1]
                
                # 鉄壁フィルター2: car_countが7以下のレースのみ残す
                if 'car_count' in df_today.columns:
                    df_today['car_count'] = pd.to_numeric(df_today['car_count'], errors='coerce')
                    df_today = df_today[df_today['car_count'] <= 7]

                if not df_today.empty:
                    df_features = generate_features_for_inference(df_today)
                    
                    # 鉄壁フィルター3: 最終確認（抽出された特徴量が正確に7人分あるか）
                    if not df_features.empty:
                        valid_race_ids = df_features.groupby('race_id', sort=False).size()
                        valid_race_ids = valid_race_ids[valid_race_ids == 7].index
                        df_features = df_features[df_features['race_id'].isin(valid_race_ids)]
                    
                    if not df_features.empty:
                        run_ai_sniper(df_features, target_date_str)
                    else:
                        print(f"[{target_date_str}] 推論可能な7車立ての選手データが抽出できなかったためスキップします。")
                else:
                    print(f"[{target_date_str}] 7車立ての新しい出走表データが存在しないため、AI推論をスキップします。")

    print("\n=== ✅ 日次バッチ処理 全工程完了 ===")

if __name__ == "__main__":
    main()
