import os
import json
import time
import math
import re
import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import itertools
import concurrent.futures
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload  # ←★追加
import io  # ←★追加
import logging
import traceback  # ← ★これを追加
import warnings
import shutil
warnings.filterwarnings("ignore")

# =============================================================================
# 1. 🌟 環境設定・定数
# =============================================================================
os.environ['TZ'] = 'Asia/Tokyo'
try: time.tzset()
except: pass

logger = logging.getLogger("Keirin_Hybrid_Sniper")
logger.setLevel(logging.INFO)
if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(sh)

JST = timezone(timedelta(hours=9), 'JST')
TODAY_OBJ = datetime.now(JST)

# APIキー・認証情報 (環境変数)
GCP_SA_CREDENTIALS = os.environ.get("GCP_SA_CREDENTIALS")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_TOKEN", "YOUR_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID", "YOUR_USER_ID")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID") 

class Config:
    DRIVE_DIR = './KeirinData' 
    MODELS_DIR = './models'
    MASTER_FILE = 'kdreams_analysis_2020-2026_master_v2.csv'  # ← 本来のマスターデータ名に修正
    MAX_WORKERS = 3 
    SLEEP_TIME = 1.0

# 🎯 聖杯ポートフォリオ条件（二刀流）
CONDITIONS_V13 = [
    {"Segment": "P2-S", "Bet_Type": "2T", "Odds_Min": 10.0, "Odds_Max": 29.9, "EV_Th": 2.5, "Limit": 3000},
    {"Segment": "P1", "Bet_Type": "2T", "Odds_Min": 30.0, "Odds_Max": 99.9, "EV_Th": 3.0, "Limit": 5000},
    {"Segment": "P1", "Bet_Type": "2T", "Odds_Min": 10.0, "Odds_Max": 29.9, "EV_Th": 2.5, "Limit": 5000}
]
CONDITIONS_V15 = [
    {"Segment": "P3", "Bet_Type": "2T", "Odds_Min": 1.0, "Odds_Max": 9.9, "EV_Th": 1.0, "Limit": 10000},
    {"Segment": "P3", "Bet_Type": "2F", "Odds_Min": 10.0, "Odds_Max": 29.9, "EV_Th": 1.2, "Limit": 3000}
]

# 各種辞書データ
KEIRIN_COORDS = {
    '函館': {"lat": 41.78, "lon": 140.75}, '青森': {"lat": 40.81, "lon": 140.71}, 'いわき平': {"lat": 37.05, "lon": 140.89},
    '弥彦': {"lat": 37.70, "lon": 138.82}, '前橋': {"lat": 36.40, "lon": 139.06}, '取手': {"lat": 35.91, "lon": 140.06},
    '宇都宮': {"lat": 36.55, "lon": 139.88}, '大宮': {"lat": 35.91, "lon": 139.63}, '西武園': {"lat": 35.76, "lon": 139.45},
    '京王閣': {"lat": 35.63, "lon": 139.53}, '立川': {"lat": 35.70, "lon": 139.42}, '松戸': {"lat": 35.80, "lon": 139.91},
    '千葉': {"lat": 35.61, "lon": 140.11}, '川崎': {"lat": 35.53, "lon": 139.71}, '平塚': {"lat": 35.32, "lon": 139.35},
    '小田原': {"lat": 35.25, "lon": 139.15}, '伊東': {"lat": 34.96, "lon": 139.09}, '静岡': {"lat": 34.97, "lon": 138.40},
    '名古屋': {"lat": 35.16, "lon": 136.87}, '岐阜': {"lat": 35.41, "lon": 136.76}, '大垣': {"lat": 35.36, "lon": 136.61},
    '豊橋': {"lat": 34.76, "lon": 137.40}, '富山': {"lat": 36.71, "lon": 137.21}, '松阪': {"lat": 34.58, "lon": 136.52},
    '四日市': {"lat": 34.98, "lon": 136.63}, '福井': {"lat": 36.07, "lon": 136.22}, '奈良': {"lat": 34.69, "lon": 135.79},
    '向日町': {"lat": 34.94, "lon": 135.71}, '和歌山': {"lat": 34.22, "lon": 135.16}, '岸和田': {"lat": 34.46, "lon": 135.38},
    '玉野': {"lat": 34.49, "lon": 133.94}, '広島': {"lat": 34.37, "lon": 132.46}, '防府': {"lat": 34.05, "lon": 131.57},
    '高松': {"lat": 34.34, "lon": 134.07}, '小松島': {"lat": 34.00, "lon": 134.59}, '高知': {"lat": 33.55, "lon": 133.55},
    '松山': {"lat": 33.82, "lon": 132.74}, '小倉': {"lat": 33.88, "lon": 130.88}, '久留米': {"lat": 33.30, "lon": 130.53},
    '武雄': {"lat": 33.19, "lon": 129.99}, '佐世保': {"lat": 33.16, "lon": 129.72}, '別府': {"lat": 33.30, "lon": 131.50},
    '熊本': {"lat": 32.80, "lon": 130.73}
}
JYO_TO_STRAIGHT = {'函館': 51.3, '青森': 58.9, 'いわき平': 62.7, '弥彦': 63.1, '前橋': 46.7, '取手': 54.8, '宇都宮': 63.3, '大宮': 66.7, '西武園': 47.6, '京王閣': 51.5, '立川': 58.0, '松戸': 38.2, '千葉': 43.0, '川崎': 58.0, '平塚': 54.2, '小田原': 36.1, '伊東': 46.6, '静岡': 37.3, '名古屋': 58.8, '岐阜': 59.3, '大垣': 56.0, '豊橋': 60.0, '富山': 43.0, '松阪': 61.5, '四日市': 62.4, '福井': 52.8, '奈良': 38.0, '向日町': 47.3, '和歌山': 59.9, '岸和田': 56.7, '玉野': 47.9, '広島': 57.9, '防府': 42.5, '高松': 56.0, '小松島': 55.5, '高知': 52.0, '松山': 58.6, '小倉': 56.9, '久留米': 50.7, '武雄': 64.4, '佐世保': 40.2, '別府': 59.9, '熊本': 60.6}
JYO_TO_PREF = {
    '函館': '北海道', '青森': '青森', 'いわき平': '福島', '弥彦': '新潟', '前橋': '群馬', '取手': '茨城', '宇都宮': '栃木', '大宮': '埼玉', '西武園': '埼玉',
    '京王閣': '東京', '立川': '東京', '松戸': '千葉', '千葉': '千葉', '川崎': '神奈川', '平塚': '神奈川', '小田原': '神奈川', '伊東': '静岡', '静岡': '静岡',
    '名古屋': '愛知', '岐阜': '岐阜', '大垣': '岐阜', '豊橋': '愛知', '富山': '富山', '松阪': '三重', '四日市': '三重', '福井': '福井', '奈良': '奈良', '向日町': '京都',
    '和歌山': '和歌山', '岸和田': '大阪', '玉野': '岡山', '広島': '広島', '防府': '山口', '高松': '香川', '小松島': '徳島', '高知': '高知', '松山': '愛媛', '小倉': '福岡',
    '久留米': '福岡', '武雄': '佐賀', '佐世保': '長崎', '別府': '大分', '熊本': '熊本'
}
BANK_MAP = {'22': '335', '31': '333', '36': '333', '37': '333', '46': '333', '53': '333', '63': '333', '25': '500', '26': '500', '74': '500'}
PREFECTURES = ["北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島", "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川", "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜", "静岡", "愛知", "三重", "滋賀", "京都", "大阪", "兵庫", "奈良", "和歌山", "鳥取", "島根", "岡山", "広島", "山口", "徳島", "香川", "愛媛", "高知", "福岡", "佐賀", "長崎", "熊本", "大分", "宮崎", "鹿児島", "沖縄"]

# =============================================================================
# 2. 🌤️ 気象アンサンブル・パイプライン (当日の予測用)
# =============================================================================
WEATHER_CACHE = {}
# 💡 V10用追加: 安定したAPI通信のためのセッション（Chromeブラウザに偽装）
HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})

# 💡 追加：全レースの判定結果を格納するグローバル辞書
JUDGMENT_REPORT = {}

def map_wmo_to_keirin_code(wmo_code):
    if wmo_code == 0: return 0 
    elif 1 <= wmo_code <= 48: return 1 
    elif 50 <= wmo_code <= 69 or 80 <= wmo_code <= 84: return 2 
    elif 70 <= wmo_code <= 79 or 85 <= wmo_code <= 99: return 3 
    return 1

def get_target_hour_index(target_time_str):
    try: return int(str(target_time_str).split(':')[0])
    except: return 15

def fetch_weather_jma_and_om(place_name, target_time_str):
    target_hour = get_target_hour_index(target_time_str)
    cache_key_full = f"JMA_OM_FULL_{place_name}"
    
    if cache_key_full not in WEATHER_CACHE:
        coords = KEIRIN_COORDS.get(place_name)
        if not coords: return None, None, None
            
        lat, lon = coords["lat"], coords["lon"]
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,weather_code&timezone=Asia%2FTokyo&forecast_days=1&models=jma_seamless,best_match"
        
        success = False
        for attempt in range(3):
            try:
                time.sleep(1.0 if attempt == 0 else 3.0)
                resp = HTTP_SESSION.get(url, timeout=15)
                if resp.status_code == 200:
                    WEATHER_CACHE[cache_key_full] = resp.json()
                    success = True
                    break
            except: pass
        if not success: return None, None, None

    data = WEATHER_CACHE[cache_key_full]
    try:
        jma_ws = round(data['hourly']['wind_speed_10m_jma_seamless'][target_hour] / 3.6, 2)
        jma_wc = map_wmo_to_keirin_code(data['hourly']['weather_code_jma_seamless'][target_hour])
        om_ws = round(data['hourly']['wind_speed_10m_best_match'][target_hour] / 3.6, 2)
        return jma_ws, jma_wc, om_ws
    except: return None, None, None

def get_ensemble_weather(place_name, target_time_str):
    target_hour = get_target_hour_index(target_time_str)
    cache_key_log = f"LOG_{place_name}_{target_hour}"
    should_log = cache_key_log not in WEATHER_CACHE
    
    if should_log:
        logger.info(f"--- 🌤️ {place_name} ({target_time_str}) の気象データ取得 ---")
        WEATHER_CACHE[cache_key_log] = True 

    jma_ws, jma_wc, om_ws = fetch_weather_jma_and_om(place_name, target_time_str)
    
    if jma_ws is not None and om_ws is not None:
        diff = abs(jma_ws - om_ws)
        if should_log:
            logger.info(f"✅ [メインAPI] 気象庁(JMA)局地モデル ＆ Open-Meteo標準モデル の取得に成功。")
            logger.info(f"📊 比較: JMA={jma_ws}m, OM={om_ws}m (差分: {diff:.1f}m)")
            if diff > 3.0:
                logger.warning(f"🚨 予報乖離エラー: ズレが3mを超えています。レースをスキップします。")
        if diff > 3.0: return None, None, False
        return jma_ws, jma_wc, True
        
    return 0.0, 1, True

# =============================================================================
# 3. 🌐 K-Dreams スクレイパー (Colab成功コードより移植)
# =============================================================================
class KDreamsAnalysisScraper:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
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
        except: return None

    def clean_text(self, text):
        return re.sub(r'\s+', ' ', text).strip() if text else "-"

    def fetch_race_urls_daily(self, date_obj):
        date_str = date_obj.strftime('%Y/%m/%d')
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
            match = re.search(r'/racedetail/\d{2}(\d{8})(\d{2})\d{4}/?', href)
            if match:
                start_date_str = match.group(1)
                day_offset = int(match.group(2)) - 1
                try:
                    race_date_obj = datetime.strptime(start_date_str, '%Y%m%d') + timedelta(days=day_offset)
                    if race_date_obj.date() == date_obj.date(): unique_urls.add(href)
                except: pass
        return list(unique_urls)

    def _extract_race_info(self, soup, url, date_str):
        meta = {'date': date_str, 'url': url}
        try:
            title_full = self.clean_text(soup.title.text)
            meta['race_title_full'] = title_full
            meta['place_name'] = title_full.split(' ')[0].replace('競輪', '')
            match = re.search(r'/racedetail/(\d{2})\d{12}(\d{2})/?', url)
            if match:
                meta['place_code'] = match.group(1)
                meta['race_num'] = str(int(match.group(2)))
            else:
                meta['place_code'] = meta['race_num'] = "-"

            bank_len_str = BANK_MAP.get(meta['place_code'], '400')
            meta['bank_length'] = bank_len_str
            meta['round'] = "一般"
            for r in ["予選", "特選", "決勝", "準決勝", "選抜"]:
                if r in title_full: meta['round'] = r; break

            info_text = soup.text
            dist_match = re.search(r'(\d{1,2},?\d{3})\s*m', info_text)
            meta['distance'] = dist_match.group(1).replace(',', '') if dist_match else "-"
            time_match = re.search(r'(?:^|\s|発走)(\d{1,2}:\d{2})(?:$|\s|発走)', info_text)
            meta['start_time'] = time_match.group(1) if time_match else "-"

            is_midnight = "ミッドナイト" in title_full
            if not is_midnight and meta['start_time'] != "-":
                try:
                    h, m = int(meta['start_time'].split(':')[0]), int(meta['start_time'].split(':')[1])
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
                    if meta['race_type'] == "P3" or meta['race_type_detail'] in ["P2-Chal", "P1"]: dist = "1625"
                    else: dist = "2025"
                elif bl == 500:
                    dist = "1525" if meta['race_type'] == "P3" else "2025"
                meta['distance'] = dist

            meta['line_prediction'] = "-"
            dt = soup.find('dt', string=re.compile('並び予想'))
            if dt:
                dd = dt.find_next_sibling('dd')
                if dd:
                    parts = []
                    for sp in dd.find_all('span', class_='icon_p'):
                        if 'space' in sp.get('class', []): parts.append("|")
                        else:
                            txt = self.clean_text(sp.text)
                            if txt and txt != "←": parts.append(txt)
                    if parts:
                        full_line = "← " + " ".join(parts)
                        full_line = re.sub(r'(\s*\|\s*)+', ' | ', full_line).strip().rstrip('|').strip()
                        meta['line_prediction'] = full_line
        except: return None
        return meta

    def _parse_name_cell(self, raw_text):
        text = re.sub(r'\s+', ' ', raw_text).strip()
        age, grad, area, name = "", "", "", text
        match = re.search(r'/(\d{1,3})/([A-Z0-9]+)$', text)
        if match:
            age, grad = match.group(1), match.group(2)
            remaining = text[:match.start()].strip()
            clean_remaining = remaining.replace(" ", "")
            for pref in PREFECTURES:
                if clean_remaining.endswith(pref):
                    area = pref
                    chars_to_remove = list(pref)
                    temp_text = list(remaining)
                    while chars_to_remove and temp_text:
                        if temp_text[-1] == chars_to_remove[-1]:
                            chars_to_remove.pop(); temp_text.pop()
                        elif temp_text[-1] == ' ': temp_text.pop()
                        else: break
                    if not chars_to_remove: name = "".join(temp_text).strip()
                    break
        return name, area, age, grad

    def _extract_players(self, soup):
        players = {}
        target_table = None
        for t in soup.find_all('table'):
            if "競走得点" in t.text and "S" in t.text and "B" in t.text:
                target_table = t; break

        if target_table:
            for row in target_table.find_all('tr'):
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
                    'nige': safe_get(8), 'maku': safe_get(9), 'sashi': safe_get(10), 'mark': safe_get(11),
                    'win': safe_get(16), 'ren2': safe_get(17), 'ren3': safe_get(18),
                    'prev_results': ""
                }

        past_table = soup.find('table', class_=re.compile('past_racecard_table'))
        if past_table:
            for row in past_table.find_all('tr'):
                num_td = row.find('td', class_='num')
                if not num_td: continue
                car_num = self.clean_text(num_td.text)
                if car_num in players:
                    times = []
                    for li in row.find_all('li'):
                        spans = li.find_all('span')
                        if len(spans) >= 4:
                            time_str = self.clean_text(spans[3].text)
                            if re.match(r'^\d+\.\d+$', time_str): times.append(f"({time_str})")
                    players[car_num]['prev_results'] = " ".join(times)
        return players

    def _extract_results(self, base_url):
        result_url = base_url + "?pageType=showResult"
        soup = self.get_soup(result_url)
        if not soup:
            return {'payout_3tan_yen': 0, 'payout_3tan_pop': 0, 'payout_2tan_yen': 0, 'payout_2fuku_yen': 0, 'weather': "", 'wind_speed': "0.0"}, {}

        weather_str, wind_speed_str = "", "0.0"
        weather_p = soup.find('p', class_='weather')
        if weather_p:
            weather_text = weather_p.text
            w_match = re.search(r'天候.*?([晴曇雨雪])', weather_text)
            if w_match: weather_str = w_match.group(1)
            s_match = re.search(r'風速.*?([0-9\.]+)\s*m', weather_text)
            if s_match: wind_speed_str = s_match.group(1)

        # ★追加：win_combo_2tan と win_combo_2fuku を初期値として追加
        payout_res = {'payout_3tan_yen': 0, 'payout_3tan_pop': 0, 'payout_2tan_yen': 0, 'payout_2fuku_yen': 0, 'win_combo_2tan': '-', 'win_combo_2fuku': '-', 'weather': weather_str, 'wind_speed': wind_speed_str}
        try:
            refund_table = soup.find('table', class_='refund_table')
            if refund_table:
                seen_3fuku = False
                for dl in refund_table.find_all('dl', class_='cf'):
                    dt, dd = dl.find('dt'), dl.find('dd')
                    if not dt or not dd: continue
                    tk = dt.get_text(strip=True)
                    clean_str = self.clean_text(dd.get_text()).replace(',', '').replace('円', '')
                    
                    yen, pop = 0, 0
                    m = re.search(r'(\d+)\((\d+)\)', clean_str)
                    if m: yen, pop = int(m.group(1)), int(m.group(2))
                    else:
                        m_yen = re.search(r'(\d+)', clean_str)
                        if m_yen: yen = int(m_yen.group(1))

                    if re.match(r'^\d+=\d+=\d+$', tk): seen_3fuku = True
                    if re.match(r'^\d+=\d+$', tk) and not seen_3fuku: 
                        payout_res['payout_2fuku_yen'] = yen
                        payout_res['win_combo_2fuku'] = tk  # ★追加：正解の出目を記憶
                    elif re.match(r'^\d+-\d+$', tk): 
                        payout_res['payout_2tan_yen'] = yen
                        payout_res['win_combo_2tan'] = tk   # ★追加：正解の出目を記憶
                    elif re.match(r'^\d+-\d+-\d+$', tk): 
                        payout_res['payout_3tan_yen'], payout_res['payout_3tan_pop'] = yen, pop
        except: pass

        ranks = {}
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

        for i in range(1, 10):
            car_key, prefix = str(i), f"c{i}"
            exists = car_key in players
            row[f"{prefix}_existence"] = 1 if exists else 0
            if exists:
                p = players[car_key]
                row[f"{prefix}_name"], row[f"{prefix}_area"], row[f"{prefix}_age"], row[f"{prefix}_grad"] = p['name'], p['area'], p['age'], p['grad']
                row[f"{prefix}_class"], row[f"{prefix}_leg"], row[f"{prefix}_gear"] = p['class'], p['leg'], p['gear']
                row[f"{prefix}_score"], row[f"{prefix}_s"], row[f"{prefix}_win"], row[f"{prefix}_2ren"], row[f"{prefix}_3ren"], row[f"{prefix}_b"] = p['score'], p['s'], p['win'], p['ren2'], p['ren3'], p['b']
                row[f"{prefix}_kimari_nige"], row[f"{prefix}_kimari_makuri"], row[f"{prefix}_kimari_sashi"], row[f"{prefix}_kimari_mark"] = p.get('nige',0), p.get('maku',0), p.get('sashi',0), p.get('mark',0)
                row[f"{prefix}_prev_results"], row[f"{prefix}_prev2_results"], row[f"{prefix}_rank"] = p.get('prev_results', ""), "", ranks.get(car_key, "0")
            else:
                for k in ['name', 'area', 'age', 'grad', 'class', 'leg', 'gear', 'prev_results', 'prev2_results', 'rank']: row[f"{prefix}_{k}"] = ""
                for k in ['score', 's', 'win', '2ren', '3ren', 'b', 'kimari_nige', 'kimari_makuri', 'kimari_sashi', 'kimari_mark']: row[f"{prefix}_{k}"] = 0
        return row

# =============================================================================
# 4. 通知・連携機能
# =============================================================================
def send_line_broadcast(message):
    if LINE_CHANNEL_ACCESS_TOKEN in ['YOUR_TOKEN', 'TEST_TOKEN', None]:
        logger.info(f"📱 【LINE通知シミュレーション】\n{message}")
        return
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    try: requests.post(url, headers=headers, json={"messages": [{"type": "text", "text": message}]})
    except Exception as e: logger.error(f"LINE API Error: {e}")

def append_to_spreadsheet(values):
    if not GCP_SA_CREDENTIALS or not SPREADSHEET_ID: return
    try:
        creds_dict = json.loads(GCP_SA_CREDENTIALS)
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        service = build('sheets', 'v4', credentials=creds)
        service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID, range='Sheet1!A:S',  # ←★ A:R を A:S に変更
            valueInputOption='USER_ENTERED', body={'values': values}
        ).execute()
        logger.info(f"📝 スプレッドシートに {len(values)} 行追記しました。")
    except Exception as e: logger.error(f"❌ スプレッドシート書き込みエラー: {e}")

# =============================================================================
# 5. 特徴量生成 (変更なし)
# =============================================================================
def parse_line_prediction(row):
    res = {}
    for i in range(1, 10):
        res[f'c{i}_line_group'], res[f'c{i}_line_pos'], res[f'c{i}_is_leader'] = 99, 99, 0
    line_str = str(row.get('line_prediction', ''))
    if pd.isna(line_str) or line_str in ['-', '', 'nan']: return pd.Series(res)
    line_str = line_str.replace('←', '').replace(' ', '')
    for g_idx, group in enumerate(line_str.split('|'), 1):
        for pos_idx, car_str in enumerate(re.findall(r'\d', group), 1):
            car_num = int(car_str)
            if 1 <= car_num <= 9:
                res[f'c{car_num}_line_group'], res[f'c{car_num}_line_pos'], res[f'c{car_num}_is_leader'] = g_idx, pos_idx, 1 if pos_idx == 1 else 0
    return pd.Series(res)

def calc_advanced_features_daily(row):
    res = {}
    place_name = str(row.get('place_name', ''))
    res['straight_length'] = JYO_TO_STRAIGHT.get(place_name, 50.0)
    home_pref = JYO_TO_PREF.get(place_name, '不明')
    scores, line_scores = [], {}
    for i in range(1, 10):
        p = f'c{i}'
        if row.get(f'{p}_existence', 0) == 1:
            score = float(row.get(f'{p}_c_score', 0.0))
            scores.append(score)
            l_group = row.get(f'{p}_line_group', 99)
            if l_group != 99:
                if l_group not in line_scores: line_scores[l_group] = []
                line_scores[l_group].append(score)
    race_mean = np.mean(scores) if scores else 0
    race_std = np.std(scores) if scores and len(scores) > 1 else 1.0
    if race_std == 0: race_std = 1.0
    line_mean_scores = {k: np.mean(v) for k, v in line_scores.items()}
    for i in range(1, 10):
        p = f'c{i}'
        if row.get(f'{p}_existence', 0) == 1:
            score = float(row.get(f'{p}_c_score', 0.0))
            res[f'{p}_score_z'] = (score - race_mean) / race_std
            l_group = row.get(f'{p}_line_group', 99)
            res[f'{p}_line_score_mean'] = line_mean_scores[l_group] if l_group != 99 and l_group in line_mean_scores else score
            res[f'{p}_is_home'] = 1 if str(row.get(f'{p}_area', '')) == home_pref else 0
        else: res[f'{p}_score_z'], res[f'{p}_line_score_mean'], res[f'{p}_is_home'] = 0.0, 0.0, 0
    return pd.Series(res)

def preprocess_and_feature_engineering(df_master, df_today_raw):
    logger.info("⚙️ 特徴量生成（時系列・ライン・偏差値・気象）を開始...")
    df_all = pd.concat([df_master, df_today_raw], ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'], errors='coerce')
    today_date = pd.to_datetime(TODAY_OBJ.date())
    mask_today = df_all['date'] >= today_date
    df_all['is_weather_stable'] = 1 
    
    for idx, row in df_all[mask_today].iterrows():
        place = row['place_name']
        rnum = int(row['race_num'])
        if place not in JUDGMENT_REPORT: JUDGMENT_REPORT[place] = {}
        
        ws, wc, is_stable = get_ensemble_weather(place, row.get('start_time', '15:00'))
        if not is_stable: 
            df_all.at[idx, 'is_weather_stable'] = 0 
            # 💡追加：気象によるスキップを記録
            JUDGMENT_REPORT[place][rnum] = {'cat': row.get('race_type_detail', '不明'), 'reason': "❌ 見送 (理由: ⏭️ 気象不安定により除外)"}
        else:
            if ws is not None: df_all.at[idx, 'wind_speed'] = ws
            if wc is not None: df_all.at[idx, 'weather_code'] = wc

    records = []
    for i in range(1, 10):
        p = f'c{i}'
        if f'{p}_existence' not in df_all.columns: continue
        mask = df_all[f'{p}_existence'] == 1
        temp = df_all.loc[mask, ['race_id', 'date', 'place_code', f'{p}_name', f'{p}_rank']].copy()
        temp.columns = ['race_id', 'date', 'place_code', 'player_name', 'rank']
        temp['car_num'] = i
        temp['rank_num'] = pd.to_numeric(temp['rank'], errors='coerce').fillna(99)
        records.append(temp)
    
    if records:
        df_long = pd.concat(records, ignore_index=True).sort_values(['player_name', 'date']).reset_index(drop=True)
        df_long['prev_date'] = df_long.groupby('player_name')['date'].shift(1)
        df_long['prev_place'] = df_long.groupby('player_name')['place_code'].shift(1)
        df_long['prev_rank'] = df_long.groupby('player_name')['rank_num'].shift(1).fillna(99)
        df_long['days_since_last_race'] = (df_long['date'] - df_long['prev_date']).dt.days.fillna(30.0)
        df_long['is_same_series'] = ((df_long['place_code'] == df_long['prev_place']) & (df_long['days_since_last_race'] <= 3)).astype(int)
        df_long['series_prev_rank'] = np.where(df_long['is_same_series'] == 1, df_long['prev_rank'], 99)
        pivot_df = df_long.pivot_table(index='race_id', columns='car_num', values=['days_since_last_race', 'series_prev_rank'])
        pivot_df.columns = [f"c{col[1]}_{'days_since_last' if col[0] == 'days_since_last_race' else 'series_prev_rank'}" for col in pivot_df.columns]
        df_all = df_all.merge(pivot_df, on='race_id', how='left')

        # 💡 ここから下の4行を追加してください：7車立てのみの日でもエラーにならないよう強制補完
        for i in range(1, 10):
            if f'c{i}_days_since_last' not in df_all.columns: df_all[f'c{i}_days_since_last'] = 30.0
            if f'c{i}_series_prev_rank' not in df_all.columns: df_all[f'c{i}_series_prev_rank'] = 99.0
        # 💡 追加ここまで

    df_all['bank_length_num'] = pd.to_numeric(df_all['bank_length'], errors='coerce').fillna(400.0)
    df_all['distance_num'] = pd.to_numeric(df_all['distance'], errors='coerce').fillna(2025.0)
    df_all['wind_speed'] = pd.to_numeric(df_all.get('wind_speed', 0), errors='coerce').fillna(0.0)
    df_all['weather_code'] = pd.to_numeric(df_all.get('weather_code', 0), errors='coerce').fillna(0).astype(int)
    df_all['place_code_num'] = pd.to_numeric(df_all['place_code'], errors='coerce').fillna(0).astype(int)
    df_all['car_count'] = pd.to_numeric(df_all.get('car_count', 7), errors='coerce').fillna(7)
    df_all['c1_is_9car_race'] = (df_all['car_count'] == 9).astype(int)

    df_lines = df_all.apply(parse_line_prediction, axis=1)
    df_all = pd.concat([df_all, df_lines], axis=1)

    for i in range(1, 10):
        p = f"c{i}"
        mask_exists = df_all[f'{p}_existence'] == 1
        
        if f'{p}_avg_3furlong' not in df_all.columns: 
            df_all[f'{p}_avg_3furlong'] = np.nan
            
        # 文字列 "(11.5) (12.0)" 等から数値を抽出して平均を出す。無い場合は11.8
        df_all.loc[mask_exists, f'{p}_avg_3furlong'] = df_all.loc[mask_exists, f'{p}_prev_results'].astype(str).apply(
            lambda x: np.mean([float(t) for t in re.findall(r'\d+\.\d+', x)]) if pd.notna(x) and re.findall(r'\d+\.\d+', x) else 11.8
        )
        df_all.loc[~mask_exists, f'{p}_avg_3furlong'] = 0.0

        for col in ['score', 'win', '2ren', 'b', 'kimari_nige', 'kimari_makuri', 'kimari_sashi', 'kimari_mark', 'age', 'grad']:
            col_name_in = f'{p}_{col}'
            col_name_out = f'{p}_c_score' if col == 'score' else (f'{p}_grad_num' if col == 'grad' else col_name_in)
            df_all[col_name_out] = pd.to_numeric(df_all.get(col_name_in, 0), errors='coerce').fillna(0.0)
            df_all.loc[~mask_exists, col_name_out] = 0.0
        df_all[f'{p}_bank_nige_cross'] = df_all['bank_length_num'] * df_all[f'{p}_kimari_nige']

    for col in ['race_type_detail', 'round']:
        if col in df_all.columns: df_all[f'{col}_code'] = df_all[col].astype('category').cat.codes
    for i in range(1, 10):
        for cat_col in ['class', 'leg', 'area']:
            col_name = f'c{i}_{cat_col}'
            if col_name in df_all.columns:
                df_all[f'{col_name}_code'] = df_all[col_name].astype('category').cat.codes
                df_all.loc[df_all[f'c{i}_existence'] != 1, f'{col_name}_code'] = -1

    df_advanced = df_all.apply(calc_advanced_features_daily, axis=1)
    df_all = pd.concat([df_all, df_advanced], axis=1)
    return df_all[df_all['date'] >= today_date].copy()

# =============================================================================
# 6. 推論 ＆ スナイプ (型変換・バグフィックス版)
# =============================================================================
def predict_and_snipe(df_today, today_str):
    logger.info(f"\n=== ⚔️ 二刀流AI スナイプ推論 ({today_str}) ===")
    try:
        v13_1st = {i: joblib.load(os.path.join(Config.MODELS_DIR, f'model_c{i}_1st_v13.pkl')) for i in range(1, 10)}
        v15_1st = {i: joblib.load(os.path.join(Config.MODELS_DIR, f'model_c{i}_1st_v15.pkl')) for i in range(1, 10)}
        v15_2nd = {i: joblib.load(os.path.join(Config.MODELS_DIR, f'model_c{i}_2nd_v15.pkl')) for i in range(1, 10)}
        odds_v13_7 = joblib.load(os.path.join(Config.MODELS_DIR, 'LGBMRegressor_7car_v13.pkl'))
        odds_v13_9 = joblib.load(os.path.join(Config.MODELS_DIR, 'LGBMRegressor_9car_v13.pkl'))
        odds_v15_g = joblib.load(os.path.join(Config.MODELS_DIR, 'LGBMRegressor_girls_v15.pkl'))
        meta_v13 = joblib.load(os.path.join(Config.MODELS_DIR, 'features_meta_v13.pkl'))
        meta_v15 = joblib.load(os.path.join(Config.MODELS_DIR, 'features_meta_v15.pkl'))
    except Exception as e:
        logger.error(f"⚠️ モデル読込エラー: {e}")
        return

    odds_features_v13 = ['Ticket_True_Prob', 'is_2F', 'c1_c_score', 'c2_c_score', 'score_diff_1_2', 'c1_avg_3furlong', 'c2_avg_3furlong', 'c1_kimari_nige', 'c2_kimari_sashi', 'c1_is_leader', 'c2_is_leader', 'is_same_line', 'c1_bank_length']
    odds_features_v15 = odds_features_v13 + ['straight_length', 'weather_code', 'wind_speed', 'c1_days_since_last', 'c2_days_since_last', 'c1_series_prev_rank', 'c2_series_prev_rank']

    message_lines = [f"📅 {today_str} 聖杯ポートフォリオ指令\n"]
    sheet_data = []
    hit_count = 0
    max_ev_today = 0.0 

    # 🚨 追加：【勝率モデル用】強制数値・カテゴリキャスト関数
    def prepare_win_features(row, meta):
        X = row[meta['features']].to_frame().T
        cat_cols = meta.get('cat_features', [])
        for col in X.columns:
            if col in cat_cols:
                # カテゴリ変数は整数にしてから category 型に変換（LightGBMのクラッシュを防ぐ）
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1).astype(int).astype('category')
            else:
                # 数値変数は float 型
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
        return X

    # 🚨 追加：【オッズモデル用】専用キャスト関数
    def prepare_odds_features(odds_dict, features_list, is_v15=False):
        df_odds = pd.DataFrame([odds_dict])[features_list]
        for col in df_odds.columns:
            df_odds[col] = pd.to_numeric(df_odds[col], errors='coerce').fillna(0.0)
        
        # V15オッズモデルのみ、特定の3カラムをcategory型に戻す
        if is_v15:
            cat_cols = ['weather_code', 'c1_series_prev_rank', 'c2_series_prev_rank']
            for col in cat_cols:
                if col in df_odds.columns:
                    df_odds[col] = df_odds[col].astype('category')
        return df_odds

    for idx, row in df_today.iterrows():
        place = row['place_name']
        rnum = int(row['race_num'])
        r_type = row['race_type_detail']
        if place not in JUDGMENT_REPORT: JUDGMENT_REPORT[place] = {}

        if row.get('is_weather_stable', 1) == 0: continue
        
        is_9car = int(row.get('c1_is_9car_race', 0))
        cars = [i for i in range(1, 10) if row.get(f'c{i}_existence', 0) == 1]
        if len(cars) < 2: 
            JUDGMENT_REPORT[place][rnum] = {'cat': r_type, 'reason': "❌ 見送 (理由: ⚠️ 出走車数不足)"}
            continue
        
        race_hit_reasons = []
        
        # 👑 【V15 ガールズ (P3)】
        if r_type == 'P3':
            X_v15 = prepare_win_features(row, meta_v15) # 安全に変換
            for c1, c2 in itertools.permutations(cars, 2):
                try:
                    p1_1st, p1_2nd = v15_1st[c1].predict(X_v15)[0], v15_2nd[c1].predict(X_v15)[0]
                    p2_1st, p2_2nd = v15_1st[c2].predict(X_v15)[0], v15_2nd[c2].predict(X_v15)[0]
                    prob_2t = min(1.0, p1_1st * p2_2nd)
                    prob_2f = min(1.0, (p1_1st * p2_2nd) + (p2_1st * p1_2nd))
                    
                    base_odds_dict = {
                        'Ticket_True_Prob': prob_2t, 'is_2F': 0, 'c1_c_score': row.get(f'c{c1}_c_score',0), 'c2_c_score': row.get(f'c{c2}_c_score',0), 
                        'score_diff_1_2': row.get(f'c{c1}_c_score',0)-row.get(f'c{c2}_c_score',0), 'c1_avg_3furlong': row.get(f'c{c1}_avg_3furlong',0), 
                        'c2_avg_3furlong': row.get(f'c{c2}_avg_3furlong',0), 'c1_kimari_nige': row.get(f'c{c1}_kimari_nige',0), 
                        'c2_kimari_sashi': row.get(f'c{c2}_kimari_sashi',0), 'c1_is_leader': row.get(f'c{c1}_is_leader',0), 
                        'c2_is_leader': row.get(f'c{c2}_is_leader',0), 'is_same_line': 0, 'c1_bank_length': row.get('bank_length_num',400), 
                        'straight_length': row.get('straight_length',50.0), 'weather_code': row.get('weather_code',0), 
                        'wind_speed': row.get('wind_speed',0.0), 'c1_days_since_last': row.get(f'c{c1}_days_since_last',30.0), 
                        'c2_days_since_last': row.get(f'c{c2}_days_since_last',30.0), 'c1_series_prev_rank': row.get(f'c{c1}_series_prev_rank',99.0), 
                        'c2_series_prev_rank': row.get(f'c{c2}_series_prev_rank',99.0)
                    }
                    
                    # 2車単オッズ計算
                    odds_df_2t = prepare_odds_features(base_odds_dict, odds_features_v15, is_v15=True)
                    pred_odds_2t = np.expm1(odds_v15_g.predict(odds_df_2t)[0])
                    ev_2t = prob_2t * pred_odds_2t
                    if ev_2t > max_ev_today: max_ev_today = ev_2t
                    
                    for cond in CONDITIONS_V15:
                        if cond['Bet_Type'] == '2T' and cond['Odds_Min'] <= pred_odds_2t <= cond['Odds_Max'] and ev_2t >= cond['EV_Th']:
                            message_lines.extend([f"👧【P3 ガールズ】{row['place_name']}{row['race_num']}R", f" 🎯 2車単 {c1}-{c2} | 予測オッズ {pred_odds_2t:.1f}倍 | EV {ev_2t:.2f}", f" 💰 上限目安: {cond['Limit']}円\n"])
                            hit_count += 1
                            sheet_data.append([TODAY_OBJ.strftime('%Y/%m/%d'), row.get('start_time',''), "V15", row['place_name'], row['race_num'], "P3", "2単", f"{c1}-{c2}", f"{prob_2t*100:.1f}%", round(pred_odds_2t, 1), round(ev_2t, 2), row.get('weather_code',0), row.get('wind_speed',0.0), cond['Limit'], "", "", "", "", ""])
                            race_hit_reasons.append(f"2単 {c1}-{c2}")
                    
                    # 2車複オッズ計算 (c1 < c2 のみ)
                    if c1 < c2:
                        base_odds_dict['Ticket_True_Prob'] = prob_2f
                        base_odds_dict['is_2F'] = 1
                        odds_df_2f = prepare_odds_features(base_odds_dict, odds_features_v15, is_v15=True)
                        pred_odds_2f = np.expm1(odds_v15_g.predict(odds_df_2f)[0])
                        ev_2f = prob_2f * pred_odds_2f
                        if ev_2f > max_ev_today: max_ev_today = ev_2f
                        
                        for cond in CONDITIONS_V15:
                            if cond['Bet_Type'] == '2F' and cond['Odds_Min'] <= pred_odds_2f <= cond['Odds_Max'] and ev_2f >= cond['EV_Th']:
                                message_lines.extend([f"👧【P3 ガールズ】{row['place_name']}{row['race_num']}R", f" 🛡️ 2車複 {c1}={c2} | 予測オッズ {pred_odds_2f:.1f}倍 | EV {ev_2f:.2f}", f" 💰 上限目安: {cond['Limit']}円\n"])
                                hit_count += 1
                                sheet_data.append([TODAY_OBJ.strftime('%Y/%m/%d'), row.get('start_time',''), "V15", row['place_name'], row['race_num'], "P3", "2複", f"{c1}={c2}", f"{prob_2f*100:.1f}%", round(pred_odds_2f, 1), round(ev_2f, 2), row.get('weather_code',0), row.get('wind_speed',0.0), cond['Limit'], "", "", "", "", ""])
                                race_hit_reasons.append(f"2複 {c1}={c2}")
                except Exception as e:
                    logger.error(f"❌ V15推論エラー: {e}\n{traceback.format_exc()}")

        # 🗡️ 【V13 男子戦 (P1, P2-S)】
        elif r_type in ['P2-S', 'P1']:
            X_v13 = prepare_win_features(row, meta_v13) # 安全に変換
            for c1, c2 in itertools.permutations(cars, 2):
                try:
                    p1, p2 = v13_1st[c1].predict(X_v13)[0], v13_1st[c2].predict(X_v13)[0]
                    prob_2t = p1 * (p2 / (1.0 - p1)) if p1 < 1.0 else 0
                    is_same_line = 1 if row.get(f'c{c1}_line_group', 99) == row.get(f'c{c2}_line_group', 99) and row.get(f'c{c1}_line_group', 99)!=99 else 0
                    
                    base_odds_dict = {
                        'Ticket_True_Prob': prob_2t, 'is_2F': 0, 'c1_c_score': row.get(f'c{c1}_c_score',0), 'c2_c_score': row.get(f'c{c2}_c_score',0), 
                        'score_diff_1_2': row.get(f'c{c1}_c_score',0)-row.get(f'c{c2}_c_score',0), 'c1_avg_3furlong': row.get(f'c{c1}_avg_3furlong',0), 
                        'c2_avg_3furlong': row.get(f'c{c2}_avg_3furlong',0), 'c1_kimari_nige': row.get(f'c{c1}_kimari_nige',0), 
                        'c2_kimari_sashi': row.get(f'c{c2}_kimari_sashi',0), 'c1_is_leader': row.get(f'c{c1}_is_leader',0), 
                        'c2_is_leader': row.get(f'c{c2}_is_leader',0), 'is_same_line': is_same_line, 'c1_bank_length': row.get('bank_length_num',400)
                    }
                    
                    odds_df_v13 = prepare_odds_features(base_odds_dict, odds_features_v13, is_v15=False)
                    odds_model = odds_v13_9 if is_9car == 1 else odds_v13_7
                    pred_odds = np.expm1(odds_model.predict(odds_df_v13)[0])
                    ev = prob_2t * pred_odds
                    if ev > max_ev_today: max_ev_today = ev
                    
                    for cond in CONDITIONS_V13:
                        if cond['Segment'] == r_type and cond['Odds_Min'] <= pred_odds <= cond['Odds_Max'] and ev >= cond['EV_Th']:
                            message_lines.extend([f"🚴‍♂️【{r_type} 男子】{row['place_name']}{row['race_num']}R", f" 🎯 2車単 {c1}-{c2} | 予測オッズ {pred_odds:.1f}倍 | EV {ev:.2f}", f" 💰 上限目安: {cond['Limit']}円\n"])
                            hit_count += 1
                            sheet_data.append([TODAY_OBJ.strftime('%Y/%m/%d'), row.get('start_time',''), "V13", row['place_name'], row['race_num'], r_type, "2単", f"{c1}-{c2}", f"{prob_2t*100:.1f}%", round(pred_odds, 1), round(ev, 2), row.get('weather_code',0), row.get('wind_speed',0.0), cond['Limit'], "", "", "", "", ""])
                            race_hit_reasons.append(f"2単 {c1}-{c2} (EV:{ev:.2f})")
                except Exception as e:
                    logger.error(f"❌ V13推論エラー: {e}\n{traceback.format_exc()}")
                
        if race_hit_reasons:
            JUDGMENT_REPORT[place][rnum] = {'cat': r_type, 'reason': f"✅ 買い (理由: ✅ 条件合致 [{', '.join(race_hit_reasons)}])"}
        else:
            JUDGMENT_REPORT[place][rnum] = {'cat': r_type, 'reason': "❌ 見送 (理由: ❌ 期待値・オッズ条件未達)"}

    # =========================================================================
    logger.info("📊 === 競輪二刀流AI 1レースごとの判定レポート ===")
    for place in sorted(JUDGMENT_REPORT.keys()):
        races = JUDGMENT_REPORT[place]
        logger.info(f"🚴‍♂️ {place} - {len(races)}レース分析")
        for rn in sorted(races.keys()):
            r = races[rn]
            logger.info(f"   {rn:>2}R: [{r['cat']}] -> {r['reason']}")
    logger.info("======================================")
    
    logger.info(f"💡 【デバッグ情報】本日の全パターンのうち、最大EVは {max_ev_today:.2f} でした。")

    if hit_count == 0: message_lines.append("本日は聖杯ポートフォリオに合致する「黄金の買い目」はありませんでした。資金を温存してください ☕")
    
    if sheet_data: append_to_spreadsheet(sheet_data)
    send_line_broadcast("\n".join(message_lines))
    logger.info(f">> ✅ 全 {hit_count} 件のスナイプ指令を送信しました！")

    # 既存のLINE送信とスプレッドシート書き込みへ続く...
    if hit_count == 0: message_lines.append("本日は聖杯ポートフォリオに合致する「黄金の買い目」はありませんでした。資金を温存してください ☕")
    
    # ↓これを追加（シートにデータがあれば書き込む）
    if sheet_data:
        append_to_spreadsheet(sheet_data)

    send_line_broadcast("\n".join(message_lines))
    logger.info(f">> ✅ 全 {hit_count} 件のスナイプ指令を送信しました！")

# ←★ここから追加：Driveへアップロードするための関数
def upload_to_drive(file_path, file_name):
    """Google Drive上の既存ファイルを検索し、上書き（または新規作成）する"""
    if not GCP_SA_CREDENTIALS: return
    try:
        creds_dict = json.loads(GCP_SA_CREDENTIALS)
        SCOPES = ['https://www.googleapis.com/auth/drive']
        creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)

        # 1. 既存の同名ファイルを検索
        query = f"name = '{file_name}' and trashed = false"
        results = service.files().list(q=query, fields="files(id)").execute()
        files = results.get('files', [])

        media = MediaFileUpload(file_path, mimetype='text/csv')

        if files:
            file_id = files[0]['id']
            service.files().update(fileId=file_id, media_body=media).execute()
            logger.info(f"✅ Google Driveの既存ファイルを更新しました (ID: {file_id})")
        else:
            file_metadata = {'name': file_name}
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            logger.info(f"✅ Google Driveに新規ファイルを作成しました (ID: {file.get('id')})")
    except Exception as e:
        logger.error(f"❌ Google Driveアップロードエラー: {e}")
# ←★ここまで追加
def download_from_drive(file_path, file_name):
    """Google Driveから最新のマスターデータをダウンロードする"""
    if not GCP_SA_CREDENTIALS: return False
    try:
        creds_dict = json.loads(GCP_SA_CREDENTIALS)
        SCOPES = ['https://www.googleapis.com/auth/drive']
        creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)

        # ファイルを検索
        query = f"name = '{file_name}' and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])

        if not files:
            logger.warning(f"⚠️ Drive上に {file_name} が見つかりません。新規作成として進行します。")
            return False

        # ダウンロード実行
        file_id = files[0]['id']
        request = service.files().get_media(fileId=file_id)
        with io.FileIO(file_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        logger.info(f"📥 Google Driveから最新のマスターデータ({file_name})をダウンロードしました！")
        return True
    except Exception as e:
        logger.error(f"❌ Driveダウンロードエラー: {e}")
        return False

def update_spreadsheet_results(yesterday_str, df_yesterday):
    """スプレッドシートの昨日の予測結果を読み込み、実際のレース結果と照合して自動更新する"""
    if not GCP_SA_CREDENTIALS or not SPREADSHEET_ID: return
    try:
        creds_dict = json.loads(GCP_SA_CREDENTIALS)
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        service = build('sheets', 'v4', credentials=creds)

        # シートのデータを全件取得 (A列〜S列)
        result = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range='Sheet1!A:S').execute() # ←★ A:Q を A:S に変更
        rows = result.get('values', [])
        if not rows: return

        update_data = []
        for i, row in enumerate(rows):
            # A列(日付)が昨日の日付と一致する行を探す
            if len(row) > 0 and row[0] == yesterday_str:
                # インデックスエラー防止のため空文字でパディング
                while len(row) < 19: row.append("") # ←★ 17 を 19 に変更

                place = row[3]
                race_num = int(row[4])
                bet_type = row[6]  # '2単' or '2複'
                combo = row[7]     # 予測した買い目 (例: '1-2')
                recommended_amt = row[13] # 推奨投資額

                # 取得した昨日のスクレイピング結果から該当レースを検索
                match = df_yesterday[(df_yesterday['place_name'] == place) & (df_yesterday['race_num'].astype(int) == race_num)]
                
                if not match.empty:
                    race_data = match.iloc[0]
                    is_hit = 0
                    confirmed_odds = 0.0

                    # 賭式ごとに正解の出目と照合
                    if bet_type == '2単':
                        if race_data.get('win_combo_2tan', '-') == combo:
                            is_hit = 1
                            confirmed_odds = race_data.get('payout_2tan_yen', 0) / 100.0
                    elif bet_type == '2複':
                        if race_data.get('win_combo_2fuku', '-') == combo:
                            is_hit = 1
                            confirmed_odds = race_data.get('payout_2fuku_yen', 0) / 100.0

                    # 更新用データの作成 (O列:実際購入額, P列:確定オッズ, Q列:的中判定)
                    # ※完全自動化のため、実際購入額(O列)には自動的に「推奨投資額(N列)」をコピー入力します
                    row_idx = i + 1
                    update_data.append({
                        'range': f'Sheet1!O{row_idx}:Q{row_idx}',
                        'values': [[recommended_amt, confirmed_odds, is_hit]]
                    })

        # 一括でスプレッドシートを更新
        if update_data:
            body = {'valueInputOption': 'USER_ENTERED', 'data': update_data}
            service.spreadsheets().values().batchUpdate(spreadsheetId=SPREADSHEET_ID, body=body).execute()
            logger.info(f"🔄 スプレッドシートの昨日の予測結果（{len(update_data)}件）の成績判定を自動入力しました！")

    except Exception as e:
        logger.error(f"❌ スプレッドシート自動判定エラー: {e}")

# =============================================================================
# 🚀 実行メインブロック (Colabカラム仕様に完全同期)
# =============================================================================
def main():

    today_dt = TODAY_OBJ.replace(hour=0, minute=0, second=0, microsecond=0)
    
    logger.info(f"[{TODAY_OBJ.strftime('%Y-%m-%d %H:%M:%S')}] 日次自動バッチ処理開始 (自動リカバリー搭載版)")
    os.makedirs(Config.DRIVE_DIR, exist_ok=True)
    full_path_master = os.path.join(Config.DRIVE_DIR, Config.MASTER_FILE)

    # 💡追加：何よりも先に、Driveから本物のマスターデータを仮想環境へ引っ張ってくる
    download_from_drive(full_path_master, Config.MASTER_FILE)

    # 万が一の破損に備え、起動時にバックアップを作成
    if os.path.exists(full_path_master):
        backup_path = full_path_master.replace('.csv', '_bak.csv')
        shutil.copy2(full_path_master, backup_path)
        logger.info(f"✅ 安全のためバックアップを作成しました: {backup_path}")

    # Colabコードの厳密なカラム定義
    first_cols = ['date', 'url', 'place_code', 'place_name', 'race_num', 'race_title_full', 'round', 'bank_length', 'distance', 'start_time', 'race_type', 'race_type_detail', 'car_count', 'payout_3tan_yen', 'payout_3tan_pop', 'payout_2tan_yen', 'payout_2fuku_yen', 'line_prediction', 'weather', 'wind_speed']
    car_cols = []
    for i in range(1, 10):
        p = f"c{i}"
        car_cols.extend([f"{p}_existence", f"{p}_name", f"{p}_area", f"{p}_age", f"{p}_grad", f"{p}_class", f"{p}_leg", f"{p}_gear", f"{p}_score", f"{p}_s", f"{p}_win", f"{p}_2ren", f"{p}_3ren", f"{p}_b", f"{p}_kimari_nige", f"{p}_kimari_makuri", f"{p}_kimari_sashi", f"{p}_kimari_mark", f"{p}_prev_results", f"{p}_prev2_results", f"{p}_rank"])

    scraper = KDreamsAnalysisScraper()
    df_master = pd.read_csv(full_path_master, low_memory=False) if os.path.exists(full_path_master) else pd.DataFrame()
    df_today_raw = pd.DataFrame()

    # 💡 追加：マスターデータから「最新日」を割り出し、取得対象の日付リストを自動生成する
    today_naive = today_dt.replace(tzinfo=None)
    start_dt_naive = today_naive - timedelta(days=1) # デフォルトは昨日

    if not df_master.empty and 'date' in df_master.columns:
        try:
            # マスターデータの最新日を取得し、その「翌日」を開始日に設定
            latest_date = pd.to_datetime(df_master['date']).max()
            if latest_date.tzinfo is not None: latest_date = latest_date.tz_localize(None)
            start_dt_naive = latest_date + timedelta(days=1)
        except Exception as e:
            logger.warning(f"⚠️ 最新日の取得に失敗しました。デフォルト(昨日)から開始します: {e}")

    # 万が一、未来の日付が最新になっていた場合は今日に補正
    if start_dt_naive > today_naive: start_dt_naive = today_naive

    # 開始日から今日までの日付リストを作成
    target_dates = []
    current_dt = start_dt_naive
    while current_dt <= today_naive:
        target_dates.append(current_dt.replace(tzinfo=JST))
        current_dt += timedelta(days=1)

    # 💡 修正：ループ対象を固定リストから、動的生成した target_dates に変更
    for target_date in target_dates:
        target_date_str = target_date.strftime('%Y-%m-%d')
        logger.info(f"\n=== 📅 【 {target_date_str} 】のデータ収集を開始 ===")
        
        urls = scraper.fetch_race_urls_daily(target_date)
        if not urls: 
            logger.info(f"❌ {target_date_str} の開催情報・レースURLが取得できませんでした。")
            continue
            
        daily_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = [executor.submit(scraper.parse_one_race, u, target_date_str) for u in urls]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    res['race_id'] = f"{res['date']}_{res['place_name']}_{res['race_num']}"
                    daily_data.append(res)
        
        if not daily_data: continue
        
        # Colabロジックの厳密なカラム結合と順序づけ
        df_day = pd.DataFrame(daily_data)
        out_cols = [c for c in first_cols if c in df_day.columns] + [c for c in car_cols if c in df_day.columns]
        remaining = [c for c in df_day.columns if c not in out_cols and c != 'race_id']
        final_cols = out_cols + ['race_id'] + remaining
        df_day = df_day[final_cols]

        if target_date == today_dt:
            df_today_raw = df_day.copy()
            logger.info(f"📝 {target_date_str} の出走表（{len(df_today_raw)}レース）を取得しました。")
        else:
            if df_master.empty:
                df_day.to_csv(full_path_master, mode='w', header=True, index=False, encoding='utf-8-sig')
                df_master = df_day.copy()
            else:
                # 安全な追記（ダミーファイル等を使わず、pandasで安全に結合して上書き）
                for col in df_master.columns:
                    if col not in df_day.columns: df_day[col] = np.nan
                df_day_aligned = df_day[df_master.columns]
                df_master = pd.concat([df_master, df_day_aligned], ignore_index=True)
                
                # ★修正：古いマスターに 'race_id' が無い場合のクラッシュを防ぐ
                if 'race_id' in df_master.columns:
                    df_master.drop_duplicates(subset=['race_id'], keep='last', inplace=True)
                
                df_master.to_csv(full_path_master, mode='w', header=True, index=False, encoding='utf-8-sig')
            logger.info(f"💾 前日({target_date_str})の結果をマスターデータに追記しました。")
            
            # ★追加：ここで「昨日のAI予測結果」の自動答え合わせ＆シート書き込みを実行！
            update_spreadsheet_results(target_date.strftime('%Y/%m/%d'), df_day)
            
        time.sleep(2)

    # 本日の推論
    if not df_today_raw.empty:
        df_ready = preprocess_and_feature_engineering(df_master, df_today_raw)
        predict_and_snipe(df_ready, TODAY_OBJ.strftime('%Y/%m/%d'))
    else:
        logger.info("本日のレースデータが存在しないため、AI推論をスキップします。")

    # ←★追加：AI推論が終わった直後、最後にマスターデータをDriveに保存する
    upload_to_drive(full_path_master, Config.MASTER_FILE)

    logger.info("=== ✅ 日次バッチ処理 全工程完了 ===")

if __name__ == "__main__":
    main()
