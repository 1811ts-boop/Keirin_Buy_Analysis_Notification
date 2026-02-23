import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def sync_files_to_drive():
    print("=== 🔄 Googleドライブへの同期を開始 ===")
    
    # GitHub Secretsから認証情報とフォルダIDを読み込む
    creds_json = os.environ.get('GCP_CREDENTIALS')
    folder_id = os.environ.get('GDRIVE_FOLDER_ID')
    target_dir = './KeirinData'
    
    if not creds_json or not folder_id:
        print("❌ 認証情報またはフォルダIDが設定されていません。")
        return

    # 認証処理
    creds_dict = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=['https://www.googleapis.com/auth/drive']
    )
    service = build('drive', 'v3', credentials=creds)

    # フォルダ内の既存ファイル一覧を取得（上書きするか新規作成するか判断するため）
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false", 
        fields="files(id, name)"
    ).execute()
    existing_files = {f['name']: f['id'] for f in results.get('files', [])}

    # KeirinDataフォルダ内のCSVとpklを全てドライブへ送信
    if not os.path.exists(target_dir):
        print("送信するデータが見つかりません。")
        return

    for file_name in os.listdir(target_dir):
        if file_name.endswith('.csv') or file_name.endswith('.pkl'):
            # バックアップは同期から除外
            if 'backup' in file_name.lower(): continue
            
            file_path = os.path.join(target_dir, file_name)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024) # MB単位に変換
            
            # 🌟 追加：安全装置（マスターファイルが80MB以下なら異常とみなしてブロック）
            if 'master' in file_name.lower() and file_size_mb < 80.0:
                print(f"🚨 致命的エラー: {file_name} のサイズが異常です（{file_size_mb:.2f} MB）。")
                print("🚨 データ破損の恐れがあるため、Googleドライブへの上書きを緊急停止しました！")
                continue # アップロードをスキップして次のファイルへ
            
            media = MediaFileUpload(file_path, resumable=True)

            if file_name in existing_files:
                # 既に存在する場合は上書き（Update）
                file_id = existing_files[file_name]
                service.files().update(fileId=file_id, media_body=media).execute()
                print(f"🔄 上書き更新完了: {file_name}")
            else:
                # 存在しない場合は新規作成（Create）
                file_metadata = {'name': file_name, 'parents': [folder_id]}
                service.files().create(body=file_metadata, media_body=media).execute()
                print(f"🆕 新規保存完了: {file_name}")

    print("=== ✅ Googleドライブへの同期が完了しました ===")

if __name__ == "__main__":
    sync_files_to_drive()
