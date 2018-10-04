from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

def authenticate():
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    return drive

def upload_file_to_gdrive(file_name, drive=None):
    if drive is None:
        drive = authenticate()

    # Create & upload a file.
    uploaded = drive.CreateFile({'title': file_name})
    uploaded.SetContentFile(file_name)
    uploaded.Upload()
    print('Uploaded file with ID {}'.format(uploaded.get('id')))
    
    
def download_file_from_gdrive(file_name, id_file, drive=None):
    if drive is None:
        drive = authenticate()

    downloaded = drive.CreateFile({'id': id_file})
    # fetch content
    downloaded.FetchContent()
    # save content to file on the disk
    import shutil
    downloaded.content.seek(0)
    with open(file_name, 'wb') as f:
        shutil.copyfileobj(downloaded.content, f, length=131072)
    print("File `{}` was successfully loaded".format(file_name))