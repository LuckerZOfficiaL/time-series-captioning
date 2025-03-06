from googleapiclient.discovery import build

from google.oauth2 import service_account



# Load credentials

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

creds = service_account.Credentials.from_service_account_file('client_secret.json', scopes=SCOPES)

service = build('drive', 'v3', credentials=creds)



def download_folder(folder_id, local_path):

    page_token = None

    while True:

        results = service.files().list(q="'"+folder_id+"' in parents", 

                                        pageSize=100, 

                                        fields="nextPageToken, files(id, name, mimeType)", 

                                        pageToken=page_token).execute()

        items = results.get('files', [])

        for item in items:

            if item['mimeType'] == 'application/vnd.google-apps.folder':

                # Recursively download subfolders

                download_folder(item['id'], os.path.join(local_path, item['name']))

            else:

                # Download file

                file_id = item['id']

                file_name = item['name']

                download_file(file_id, os.path.join(local_path, file_name))



        page_token = results.get('nextPageToken', None)

        if not page_token:

            break



def download_file(file_id, local_path):

    request = service.files().get_media(fileId=file_id)

    fh = open(local_path, 'wb')

    downloader = MediaIoBaseDownload(fh, request)

    done = False

    while not done:

        status, done = downloader.next_chunk()

        print('Download progress: {0}'.format(int(100 * status.progress())))



# Replace with your folder ID

folder_id = '1kTT7PKZd2N2wF6AgZOaYuaDW00v4HNoc'

local_download_path = '/atad'

download_folder(folder_id, local_download_path)
