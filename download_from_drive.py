import os
import io
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

def download_from_drive():
    print("=== 📥 Googleドライブからデータをダウンロード ===")
    creds_json = os.environ.get('GCP_CREDENTIALS')
    folder_id = os.environ.get('GDRIVE_FOLDER_ID')
    target_dir = './KeirinData'
    os.makedirs(target_dir, exist_ok=True)

    if not creds_json or not folder_id:
        print("❌ 認証情報またはフォルダIDが設定されていません。")
        return

    # 認証処理
    creds_dict = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    service = build('drive', 'v3', credentials=creds)

    # フォルダ内のファイル一覧を取得
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name)"
    ).execute()
    
    items = results.get('files', [])
    if not items:
        print("ドライブにファイルが見つかりません。")
        return
        
    for item in items:
        file_id = item['id']
        file_name = item['name']
        # CSVとpklファイルのみダウンロード
        if file_name.endswith('.csv') or file_name.endswith('.pkl'):
            print(f"⬇️ ダウンロード中: {file_name}")
            request = service.files().get_media(fileId=file_id)
            file_path = os.path.join(target_dir, file_name)
            with io.FileIO(file_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
    print("=== ✅ ダウンロード完了 ===")

if __name__ == '__main__':
    download_from_drive()
